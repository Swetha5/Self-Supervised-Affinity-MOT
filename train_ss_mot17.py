"""
@author swetha
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
import cv2
from mot_ss_train_dataset import MOTSSTrainDataset
from config.config import config
from layer.sst import build_sst
from utils.augmentations import SSJAugmentation, collate_fn
from layer.sst_loss import SSTLoss
import time
import torchvision.utils as vutils
from utils.operation import show_circle, show_batch_circle_image
from utils.arg_parse import args


args.cyc_sampling = False
config['save_base_path'] = args.save_base_path
config['save_folder'] = args.save_folder
config['save_images_folder'] = args.save_images_folder

if not os.path.exists(args.save_base_path):
    os.mkdir(args.save_base_path)

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if not os.path.exists(args.save_images_folder):
    os.mkdir(args.save_images_folder)

sst_dim = config['sst_dim']
means = config['mean_pixel']
batch_size = args.batch_size
max_iter = args.iterations
weight_decay = args.weight_decay
gamma = args.gamma
momentum = args.momentum
print(args)

if args.tensorboard:
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(log_dir=args.save_base_path)

sst_net = build_sst('train')
net = sst_net

if args.cuda:
    net = torch.nn.DataParallel(sst_net)
    cudnn.benchmark = True

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    sst_net.load_weights(args.resume)
else:
    print('No weight initialization!!')


if args.cuda:
    net = net.cuda()


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if not args.resume:
    print('Xavier uniform weight initializing for VGG, Extras, Selector, Final Net ...')
    sst_net.vgg.apply(weights_init)
    sst_net.extras.apply(weights_init)
    sst_net.selector.apply(weights_init)
    sst_net.final_net.apply(weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)

criterion = SSTLoss(args.cuda)


def train():
    net.train()
    current_lr = config['learning_rate']
    print('Loading Dataset...')

    dataset = MOTSSTrainDataset(args.mot_root, SSJAugmentation(sst_dim, means))

    epoch_size = len(dataset) // args.batch_size
    print('Training SSJ on', dataset.dataset_name)
    config['epoch_size'] = epoch_size

    if 'learning_rate_decay_by_epoch' in config:
        stepvalues = list((config['epoch_size'] * i for i in config['learning_rate_decay_by_epoch']))
        save_weights_iteration = config['save_weight_every_epoch_num'] * config['epoch_size']
    else:
        stepvalues = (90000, 95000)
        save_weights_iteration = epoch_size

    step_index = 0
    batch_iterator = None

    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers, shuffle=True,
                                  collate_fn=collate_fn, pin_memory=False)

    for iteration in range(args.start_iter, max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
            all_epoch_loss = []
            # all_epoch_cyc_loss = []
            all_epoch_total_loss = []

        if iteration in stepvalues:
            step_index += 1
            current_lr = adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        img_pre, img_next, boxes_pre, boxes_next, labels, valid_pre, valid_next = \
            next(batch_iterator)

        if args.cuda:
            img_pre = Variable(img_pre.cuda())
            img_next = Variable(img_next.cuda())
            boxes_pre = Variable(boxes_pre.cuda())
            boxes_next = Variable(boxes_next.cuda())
            valid_pre = Variable(valid_pre.cuda(), volatile=True)
            valid_next = Variable(valid_next.cuda(), volatile=True)
            labels = Variable(labels.cuda(), volatile=True)

        else:
            img_pre = Variable(img_pre)
            img_next = Variable(img_next)
            boxes_pre = Variable(boxes_pre)
            boxes_next = Variable(boxes_next)
            valid_pre = Variable(valid_pre)
            valid_next = Variable(valid_next)
            labels = Variable(labels, volatile=True)

        # forward
        t0 = time.time()

        out, feat_pre, feat_next = net(img_pre, img_next, boxes_pre, boxes_next, valid_pre, valid_next)

        optimizer.zero_grad()
        loss_pre, loss_next, loss_similarity, loss_con, loss, accuracy_pre, accuracy_next, accuracy, predict_indexes = criterion(
            out, labels, valid_pre, valid_next)

        total_loss = loss

        total_loss.backward()
        optimizer.step()
        t1 = time.time()

        all_epoch_loss += [loss.data.cpu()]
        all_epoch_total_loss += [total_loss.data.cpu()]

        if iteration % 30 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ', ' + repr(epoch_size) + ' || epoch: %.4f ' % (
                    iteration / (float)(epoch_size)) + ' || Total Loss: %.4f ||' % (total_loss.data[0])
                  + ' || Loss: %.4f ||' % (loss.data[0])
                  , end=' ')

        if args.tensorboard:
            if len(all_epoch_total_loss) > 30:
                writer.add_scalar('data/epoch_loss', float(np.mean(all_epoch_loss)), iteration)
                writer.add_scalar('data/epoch_total_loss', float(np.mean(all_epoch_total_loss)), iteration)
            writer.add_scalar('data/learning_rate', current_lr, iteration)

            writer.add_scalar('loss/total_loss', total_loss.data.cpu(), iteration)
            writer.add_scalar('loss/loss', loss.data.cpu(), iteration)
            writer.add_scalar('loss/loss_pre', loss_pre.data.cpu(), iteration)
            writer.add_scalar('loss/loss_next', loss_next.data.cpu(), iteration)
            writer.add_scalar('loss/loss_similarity', loss_similarity.data.cpu(), iteration)
            writer.add_scalar('loss/loss_consistency', loss_con.data.cpu(), iteration)

            writer.add_scalar('accuracy/accuracy', accuracy.data.cpu(), iteration)
            writer.add_scalar('accuracy/accuracy_pre', accuracy_pre.data.cpu(), iteration)
            writer.add_scalar('accuracy/accuracy_next', accuracy_next.data.cpu(), iteration)

            # add images
            if args.send_images and iteration % 1000 == 0:
                result_image = show_batch_circle_image(img_pre, img_next, boxes_pre, boxes_next, valid_pre, valid_next,
                                                       predict_indexes, iteration)

                grid = vutils.make_grid(result_image, normalize=True)
                writer.add_image('WithLabel/ImageResult', grid, iteration)

        if iteration % save_weights_iteration == 0:
            print('Saving state, iter:', iteration)
            torch.save(sst_net.state_dict(),
                       os.path.join(args.save_folder, 'ss_mot_' + repr(iteration) + '.pth'))

    torch.save(sst_net.state_dict(), args.save_folder + 'final' + args.version + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
