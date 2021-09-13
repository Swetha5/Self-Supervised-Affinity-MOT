import argparse
import logging
import os
import pprint
import time

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import yaml
from easydict import EasyDict
#from kitti_devkit.evaluate_tracking import evaluate
from torch.utils.data import DataLoader
from tracking_model import TrackingModule
from SAMOT import build_model
#from utils.build_util import build_augmentation, build_dataset
from data_util import write_kitti_result
from train_util import AverageMeter, create_logger
from convert_to_JRDB import files_convert_to_JRDB_formate

parser = argparse.ArgumentParser(description='PyTorch mmMOT Evaluation')
#parser.add_argument('--config', default='config.yaml')
#parser.add_argument('--load-path', default='new_weights/2D_next_random/ssj300_0712_50.0.pth', type=str)
#parser.add_argument('--result-path', default="/home/aakash/Downloads/SAMOT_gpu/evaluation_results/random_next_2D_more_experiments/experiment_1/epoch_40", type=str)
#parser.add_argument('--result-path', default="weights-80/train_with_kitti_weights-80/max_sample_10/test_predictions/calculate_time", type=str)
#parser.add_argument('--result-path', default="weights-100-new/o3d_modified/exp_1/test_60_threshold_05", type=str)
parser.add_argument('--recover', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--result_sha', default='last')
parser.add_argument('--memory', action='store_true')

global args, config, best_mota
args = parser.parse_args()

with open('config.yaml') as f:
	config = yaml.load(f, Loader=yaml.FullLoader)

config = EasyDict(config['common'])
args.result_path=config.test_prediction_dir




from test_dataset import TestSequenceDataset
def main():
	#root_dir='/home/c3-0/datasets/JRDB/MOT_output_pt/test_sequences/'
	#root_dir='/home/c3-0/datasets/JRDB/MOT_output_pt/sequences/'

	print('start_creating_dataset')
	val_dataset=TestSequenceDataset(config.root_dir+'test_sequences/',label_dir=config.test_label_dir)
	print('dataset_created')




	#load_model_dir='/home/aa809504/SAMOT_gpu/new_weights/Pedestrian/experiment_1/ssj300_0712_epoch_45.0_Loss_ 0.2763.pth'
	#load_model_dir='weights-100-new/o3d_modified/exp_1/ssj300_0712_epoch_60.0_Loss_ 0.0617.pth'
	print('load_weights_from', config.load_weights)

	
	cuda_device=True
	
	sst_net = build_model('train',cuda_device)

	model=sst_net.cuda()


	model.load_state_dict(torch.load(config.load_weights))
	tracking_module = TrackingModule(model, None, None, '3D')

	cudnn.benchmark = True
	#save_path = '/home/aakash/Downloads/SAMOT_gpu'



	#logger = create_logger('global_logger', save_path + '/eval_log.txt')
	#logger.info('args: {}'.format(pprint.pformat(args)))
	#logger.info('config: {}'.format(pprint.pformat(config)))
	validate(val_dataset, tracking_module, 'last', part='val')

def validate(val_loader,
			 tracking_module,
			 step,
			 part='train',
			 fusion_list=None,
			 fuse_prob=False):
	prec = AverageMeter(0)
	rec = AverageMeter(0)
	mota = AverageMeter(0)
	motp = AverageMeter(0)
	total_lenth=0
	start_time = time.time()

	logger = logging.getLogger('global_logger')
	for i, (sequence) in enumerate(val_loader):
		logger.info('Test: [{}/{}]\tSequence ID: KITTI-{}'.format(
			i, len(val_loader), sequence.name))
		seq_loader = DataLoader(
			sequence,
			batch_size=config.batch_size,
			shuffle=False,
			num_workers=config.workers,
			pin_memory=True)
		if len(seq_loader) == 0:
			tracking_module.eval()
			logger.info('Empty Sequence ID: KITTI-{}, skip'.format(
				sequence.name))
		else:
			#if args.memory:
			#    seq_prec, seq_rec, seq_mota, seq_motp = validate_mem_seq(
			#        seq_loader, tracking_module)
			#else:
			print('sequence number ', i)
			print('validate_sequence ',sequence.name)
			print('sequence length ',len(sequence))
			total_lenth+=len(sequence)
			print(total_lenth)
			seq_prec, seq_rec, seq_mota, seq_motp = validate_seq(
				seq_loader, tracking_module)
			#print('seq_mota',seq_mota)
			#print()
			prec.update(seq_prec, 1)
			rec.update(seq_rec, 1)
			mota.update(seq_mota, 1)
			motp.update(seq_motp, 1)

		write_kitti_result(
			args.result_path,
			sequence.name,
			step,
			tracking_module.frames_id,
			tracking_module.frames_det,
			part=part)
		read_dir=args.result_path+'/'+step+'/'+part+'/'+sequence.name+'.txt'
		write_dir=args.result_path+'/'+'JRDB_formate'
		if not os.path.exists(write_dir):
			os.mkdir(write_dir)
		files_convert_to_JRDB_formate(read_dir,write_dir+'/'+sequence.name+'.txt')



	total_num = torch.Tensor([prec.count])
	logger.info(
		'* Prec: {:.3f}\tRec: {:.3f}\tMOTA: {:.3f}\tMOTP: {:.3f}\ttotal_num={}'
		.format(prec.avg, rec.avg, mota.avg, motp.avg, total_num.item()))
	#MOTA, MOTP, recall, prec, F1, fp, fn, id_switches = evaluate(
	#    step, args.result_path, part=part)

	tracking_module.train()
	#return MOTA, MOTP, recall, prec, F1, fp, fn, id_switches
	end_time=time.time()
	total_time=end_time-start_time
	time_per_sample=total_time/total_lenth
	print('time_per_sample ',time_per_sample)
	print('test_files_generated')


def validate_seq(val_loader,
				 tracking_module,
				 fusion_list=None,
				 fuse_prob=False):
	batch_time = AverageMeter(0)

	# switch to evaluate mode
	#print('sequence_length',len(val_loader))
	tracking_module.eval()

	logger = logging.getLogger('global_logger')
	end = time.time()

	with torch.no_grad():
		for i, (input, det_info, dets, det_split) in enumerate(val_loader):
			#print('det_info',det_info)
			#input = input.cuda()
			'''
			if len(det_info) > 0:
				for k, v in det_info.items():
					det_info[k] = det_info[k].cuda() if not isinstance(
						det_info[k], list) else det_info[k]
			'''

			# compute output
			aligned_ids, aligned_dets, frame_start = tracking_module.predict(
				input, det_info, dets, det_split)
			#print(aligned_ids)
			if i%100==0:
				print('samples_passed',i)

			batch_time.update(time.time() - end)
			end = time.time()
			if i % config.print_freq == 0:
				logger.info('Test Frame: [{0}/{1}]\tTime {batch_time.val:.3f}'
							'({batch_time.avg:.3f})'.format(
								i, len(val_loader), batch_time=batch_time))

	return 0, 0, 0, 0


if __name__ == '__main__':
	main()
