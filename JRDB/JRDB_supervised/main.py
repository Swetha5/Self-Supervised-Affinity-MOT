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
from layer.sst_loss import SSTLoss
#from kitti_devkit.evaluate_tracking import evaluate
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
# from models import model_entry
#from tracking_model import TrackingModule
#from utils.build_util import build_dataset
#from utils.build_util import build_augmentation
from torch.autograd import Variable
import torch.optim as optim

from SAMOT import build_model
from train_dataset import TrainDataset
from torch.utils.data.sampler import Sampler
import numpy as np
import torch


parser = argparse.ArgumentParser(description='PyTorch mmMOT Training')
parser.add_argument('--config', default='config.yaml')


def main():

	global args, config, best_mota
	args = parser.parse_args()

	with open(args.config) as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

	config = EasyDict(config['common'])
	config.save_path = os.path.dirname(args.config)
	load_previous_weights=False
	cuda_device='True'

	best_mota = 0
	load_dataset=False


	cudnn.benchmark = True
	tensorboard=True

	#sample_max_len=20
	#save_folder='weights/train_from_scratch/max_sample_20'
	#log_address='weights/train_from_scratch/max_sample_20/log'
	
	save_weights=config.save_weights
	log_address=save_weights+'/log'

	#load_model_dir='/home/aa809504/SAMOT_gpu/new_weights/Pedestrian/experiment_1/ssj300_0712_epoch_45.0_Loss_ 0.2763.pth'
	# create model
	epochs=120
	sst_net = build_model('train',cuda_device)
	

	if cuda_device:
		net=sst_net.cuda()
	else:
		net=sst_net

	optimizer=optim.Adam(net.parameters())
	if os.path.exists(os.path.join(save_weights, 'prev_state.pth')):
		prev_states = torch.load(os.path.join(save_weights, 'prev_state.pth'))
		last_iter = prev_states['iter']
		print('previous iterations found', last_iter)
		model_weights=prev_states['model_weights']
		net.load_state_dict(model_weights)
		optimizer_states=prev_states['optimizer_states']
		optimizer.load_state_dict(optimizer_states)


	else:
		last_iter = -1
		if load_previous_weights:
			net.load_state_dict(torch.load(load_model_dir))
			print('kitti_weights_loaded')
		else:
			print('training_from_scratch')


	criterion = SSTLoss(cuda_device)




	batch_size=config.batch_size
	print('laoding_Data...')
	#if os.path.exists('dataset_pt_files/train_dataset_20.pt'):

	print('builing dataset')
	train_dataset=TrainDataset(config.root_dir+'sequences/',label_dir=config.train_label_dir, sample_max_len=2)
	#torch.save(train_dataset,'dataset_pt_files/train_dataset_20.pt')

	print('length of dataset',len(train_dataset))
	

		#for iteration in range(start_iter, iterations):

	train_data_length=len(train_dataset)


		

	max_iter=epochs*train_data_length
	print(max_iter ,'max_iter')
	train_sampler = DistributedGivenIterationSampler(
		train_dataset,
		max_iter,
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


	if tensorboard:
		from tensorboardX import SummaryWriter
		if not os.path.exists(log_address):
			os.mkdir(log_address)
		writer = SummaryWriter(log_dir=log_address) 


	net.train()
	print('training..')
	epoch_start_time=time.time()
	for i, (input, det_info, det_id, det_cls,det_split) in enumerate(train_loader):
		current_step=last_iter + i+1

		#print(current_step, ' current_step')
		t0 = time.time()
		out = net(input, det_info, det_id, det_cls, det_split)
		val_num=det_cls[0].shape[1]+det_cls[1].shape[1]
		det_score=torch.zeros((1,val_num),dtype=torch.float32)
		gt_det, gt_link, gt_new, gt_end = generate_gt(det_score[0], det_cls, det_id, det_split)

		m = nn.ConstantPad2d((0, 101-gt_link[0][0].shape[1], 0, 101-gt_link[0][0].shape[0]),0 )
		gt_link=m(gt_link[0][0])
		gt_link=torch.unsqueeze(gt_link, 0)
		gt_link=torch.unsqueeze(gt_link, 0)
		labels=gt_link

		valid_pre=gt_det[:det_split[0].item()]
		m = nn.ConstantPad1d((0, 101-valid_pre.shape[0]), 0)
		valid_pre=m(valid_pre)
		valid_pre=torch.unsqueeze(valid_pre, 0)
		valid_pre=torch.unsqueeze(valid_pre, 0)
		valid_pre[0][0][100]=1

		valid_next=gt_det[det_split[0].item():]
		m = nn.ConstantPad1d((0, 101-valid_next.shape[0]), 0)
		valid_next=m(valid_next)
		valid_next=torch.unsqueeze(valid_next, 0)
		valid_next=torch.unsqueeze(valid_next, 0)
		valid_next[0][0][100]=1

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

		if current_step%100==0:
			print('Timer: %.4f sec.' % (t1 - t0))
			print('Training_Samples ' + ', ' + repr(current_step) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

		if current_step%train_data_length==0:
			actual_epoch=current_step/train_data_length
			print('epoch number: ' + repr(actual_epoch))
			writer.add_scalar('loss/loss', loss.data.cpu(), actual_epoch)
			writer.add_scalar('loss/loss_pre', loss_pre.data.cpu(),actual_epoch)
			writer.add_scalar('loss/loss_next', loss_next.data.cpu(), actual_epoch)
			writer.add_scalar('loss/loss_similarity', loss_similarity.data.cpu(), actual_epoch)

			writer.add_scalar('accuracy/accuracy', accuracy.data.cpu(), actual_epoch)
			writer.add_scalar('accuracy/accuracy_pre', accuracy_pre.data.cpu(), actual_epoch)
			writer.add_scalar('accuracy/accuracy_next', accuracy_next.data.cpu(), actual_epoch)

			for name, param in net.named_parameters():
				writer.add_histogram(name, param.clone().cpu().data.numpy(), current_step/train_data_length, bins='doane')

			epoch_end_time=time.time()

			print('time taken for an epoch', epoch_end_time-epoch_start_time)
			epoch_start_time=time.time()


		if current_step%(train_data_length*1)==0:
			print('Saving state, epoch:', actual_epoch)
			torch.save(net.state_dict(),
					   os.path.join(
						   save_weights,
						   'ssj300_0712_' +'epoch_'+ repr(actual_epoch)+ '_Loss_ %.4f' % (loss.item())+'.pth'))
		if current_step%(int(train_data_length*0.5))==0:
		#if (current_step)%10==0:
			print('saving checkpoint')
			curr_states = {
			'iter': current_step,
			'model_weights': net.state_dict(),
			'optimizer_states' : optimizer.state_dict()
	 		#save optimizer as well	
			}
			torch.save(curr_states, os.path.join(save_weights, 'prev_state.pth'))


	#torch.save(sst_net.state_dict(), save_folder + '' + 'version_v1'+ '.pth')

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




class DistributedGivenIterationSampler(Sampler):

	def __init__(self,
				 dataset,
				 total_iter,
				 batch_size,
				 world_size=None,
				 rank=None,
				 last_iter=-1):
		self.dataset = dataset
		self.total_iter = total_iter
		self.batch_size = batch_size
		self.world_size = world_size
		self.rank = rank
		self.last_iter = last_iter

		self.total_size = self.total_iter * self.batch_size

		self.indices = self.gen_new_list()
		self.call = 0

	def __iter__(self):
		if self.call == 0:
			self.call = 1
			return iter(self.indices[(self.last_iter + 1) * self.batch_size:])
		else:
			raise RuntimeError(
				"this sampler is not designed to be called more than once!!")

	def gen_new_list(self):

		np.random.seed(0)
		all_size = self.total_size * self.world_size
		origin_indices = np.arange(len(self.dataset))
		origin_indices = origin_indices[:all_size]
		num_repeat = (all_size - 1) // origin_indices.shape[0] + 1

		total_indices = []
		for i in range(num_repeat):
			total_indices.append(np.random.permutation(origin_indices))
		indices = np.concatenate(total_indices, axis=0)[:all_size]

		beg = self.total_size * self.rank
		indices = indices[beg:beg + self.total_size]

		assert len(indices) == self.total_size

		return indices

	def __len__(self):
		return self.total_size


if __name__ == '__main__':
	main()
