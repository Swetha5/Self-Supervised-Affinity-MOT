import argparse
import logging
import os
import pprint
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import yaml
from easydict import EasyDict
#from kitti_devkit.evaluate_tracking import evaluate
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
# from models import model_entry
#from tracking_model import TrackingModule
from utils.build_util import build_dataset
from utils.build_util import build_augmentation
from torch.autograd import Variable
import torch.optim as optim

from network import build_model
from layer.sst_loss import SSTLoss

from utils.data_util import write_kitti_result
from utils.train_util import (AverageMeter, DistributedGivenIterationSampler,
							  create_logger, load_state, save_checkpoint)


parser = argparse.ArgumentParser(description='PyTorch PC-DAN Self-Supervised Training')
parser.add_argument('--config', default='config.yaml')


def main():

	global args, config, best_mota
	args = parser.parse_args()

	with open(args.config) as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	config = EasyDict(config['common'])
	config.save_path = os.path.dirname(args.config)

	cuda_device='True'

	# create model
	sst_net = build_model('train',cuda_device)
	
	if cuda_device:
		net=sst_net.cuda()
	else:
		net=sst_net

	criterion = SSTLoss(cuda_device)

	optimizer=optim.Adam(net.parameters())
	# optionally resume from a checkpoint
	last_iter = -1
	best_mota = 0
	load_dataset=False

	cudnn.benchmark = True
	tensorboard=True

	if tensorboard:
		from tensorboardX import SummaryWriter
		if not os.path.exists(config.log_address):
			os.mkdir(config.log_address)
		writer = SummaryWriter(log_dir=config.log_address) 

	# Data loading code
	train_transform, valid_transform = build_augmentation(config.augmentation)

	# train
	print('laoding_Data...')
	if load_dataset:
		train_dataset=torch.load(config.pre_data_root+'/'+'train_dataset_Cars.pt')
	else:
		train_dataset = build_dataset(
			config,
			set_source='train',
			evaluate=False,
			train_transform=train_transform)
		torch.save(train_dataset, config.pre_data_root+'/'+'train_dataset_Cars.pt')

	net.train()
	batch_size=config.batch_size

	print('length of dataset',len(train_dataset))
	print('training..')

	train_sampler = DistributedGivenIterationSampler(
		train_dataset,
		config.lr_scheduler.max_iter,
		config.batch_size,
		world_size=1,
		rank=0,
		last_iter=last_iter)

	train_loader = DataLoader(
		train_dataset,
		batch_size=config.batch_size,
		shuffle=False,
		num_workers=config.workers,
		pin_memory=True,
		sampler=train_sampler)

	epoch_start_time=time.time()
	for i, (input, det_info, det_id, det_cls,det_split) in enumerate(train_loader):
		if i==0:
			print(det_info)
		t0 = time.time()
		out = net(input, det_info, det_id, det_cls, det_split)
		det_score=torch.zeros((1,input[0].shape[0]),dtype=torch.float32)
		gt_det, gt_link, gt_new, gt_end = generate_gt(det_score[0], det_cls, det_id, det_split)

		m = nn.ConstantPad2d((0, 81-gt_link[0][0].shape[1], 0, 81-gt_link[0][0].shape[0]),0 )
		gt_link=m(gt_link[0][0])
		gt_link=torch.unsqueeze(gt_link, 0)
		gt_link=torch.unsqueeze(gt_link, 0)
		labels=gt_link

		valid_pre=gt_det[:det_split[0].item()]
		m = nn.ConstantPad1d((0, 81-valid_pre.shape[0]), 0)
		valid_pre=m(valid_pre)
		valid_pre=torch.unsqueeze(valid_pre, 0)
		valid_pre=torch.unsqueeze(valid_pre, 0)
		valid_pre[0][0][80]=1

		valid_next=gt_det[det_split[0].item():]
		m = nn.ConstantPad1d((0, 81-valid_next.shape[0]), 0)
		valid_next=m(valid_next)
		valid_next=torch.unsqueeze(valid_next, 0)
		valid_next=torch.unsqueeze(valid_next, 0)
		valid_next[0][0][80]=1
		if cuda_device:
			out=out.cuda()
			valid_pre = Variable(valid_pre.cuda())
			valid_next = Variable(valid_next.cuda())
			labels = Variable(labels.cuda())

		else:
			valid_pre = Variable(valid_pre)
			valid_next = Variable(valid_next)
			labels = Variable(labels)


		optimizer.zero_grad()
		loss_pre, loss_next, loss_similarity, loss, accuracy_pre, accuracy_next, accuracy, predict_indexes = criterion(out, labels, valid_pre, valid_next)

		loss.backward()
		optimizer.step()
		t1 = time.time()

		if i%100== 0:
			print('Timer: %.4f sec.' % (t1 - t0))
			print('Training_Samples ' + ', ' + repr(i) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

		if i%3355==0:
			actual_epoch=i/3355
			print('epoch number: ' + repr(actual_epoch))
			writer.add_scalar('loss/loss', loss.data.cpu(), actual_epoch)
			writer.add_scalar('loss/loss_pre', loss_pre.data.cpu(),actual_epoch)
			writer.add_scalar('loss/loss_next', loss_next.data.cpu(), actual_epoch)
			writer.add_scalar('loss/loss_similarity', loss_similarity.data.cpu(), actual_epoch)

			writer.add_scalar('accuracy/accuracy', accuracy.data.cpu(), actual_epoch)
			writer.add_scalar('accuracy/accuracy_pre', accuracy_pre.data.cpu(), actual_epoch)
			writer.add_scalar('accuracy/accuracy_next', accuracy_next.data.cpu(), actual_epoch)

			for name, param in net.named_parameters():
				writer.add_histogram(name, param.clone().cpu().data.numpy(), i/3000, bins='doane')

			epoch_end_time=time.time()

			print('time taken for an epoch', epoch_end_time-epoch_start_time)
			epoch_start_time=time.time()

		if i%(3355*1)==0:
			print('Saving state, epoch:', i)
			torch.save(sst_net.state_dict(),
					   os.path.join(
						   config.save_folder, 'epoch_'+ repr(i)+ '_Loss_ %.4f' % (loss.item())+'.pth'))

	torch.save(sst_net.state_dict(), config.save_folder + '.pth')

def adjust_learning_rate(optimizer, gamma, step):
	"""Sets the learning rate to the initial LR decayed by 10 at every specified step
	# Adapted from PyTorch Imagenet example:
	# https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""
	learning_rate=5e-3
	lr = learning_rate * (gamma ** (step))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr

def generate_gt(det_score, det_cls, det_id, det_split):
	gt_det = det_score.new_zeros(det_score.size())
	gt_new = det_score.new_zeros(det_score.size())
	gt_end = det_score.new_zeros(det_score.size())
	gt_link = []
	det_start_idx = 0

	for i in range(len(det_split)):
		det_curr_num = det_split[i]  # current frame i has det_i detects
		if i != len(det_split) - 1:
			link_matrix = det_score.new_zeros(
				(1, det_curr_num, det_split[i + 1]))
		# Assign the score, according to eq1
		for j in range(det_curr_num):
			curr_det_idx = det_start_idx + j
			# g_det
			if det_cls[i][0][j] == 1:
				gt_det[curr_det_idx] = 1  # positive
			else:
				continue

			# g_link for successor frame
			if i == len(det_split) - 1:
				# end det at last frame
				gt_end[curr_det_idx] = 1
			else:
				matched = False
				det_next_num = det_split[i + 1]
				for k in range(det_next_num):
					if det_id[i][0][j] == det_id[i + 1][0][k]:
						link_matrix[0][j][k] = 1
						matched = True
						break
				if not matched:
					# no successor means an end det
					gt_end[curr_det_idx] = 1

			if i == 0:
				# new det at first frame
				gt_new[curr_det_idx] = 1
			else:
				# look prev
				matched = False
				det_prev_num = det_split[i - 1]
				for k in range(det_prev_num):
					if det_id[i][0][j] == det_id[i - 1][0][k]:
						# have been matched during search in
						# previous frame, no need to assign
						matched = True
						break
				if not matched:
					gt_new[curr_det_idx] = 1

		det_start_idx += det_curr_num
		if i != len(det_split) - 1:
			gt_link.append(link_matrix)

	return gt_det, gt_link, gt_new, gt_end




if __name__ == '__main__':
	main()
