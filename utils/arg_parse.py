"""
@project: ss-mot
@author: swetha
"""

import argparse
from config.config import config

str2bool = lambda v: v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Self supervised Cyc MOT')
parser.add_argument('--version', default='v1', help='current version')
parser.add_argument('--basenet', help='pretrained base model weights')
parser.add_argument('--matching_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=config['batch_size'], type=int, help='Batch size for training')
parser.add_argument('--resume', default=config['resume'], type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=config['num_workers'], type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=config['iterations'], type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=config['start_iter'], type=int,
                    help='Begin counting iterations starting from this value (used with resume)')
parser.add_argument('--cuda', default=config['cuda'], type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', default=config['learning_rate'], type=float,
                    help='initial learning rate')
parser.add_argument('--epsilon', default=0.3, type=float,
                    help='epsilon parameter for temperature')
parser.add_argument('--margin', default=0.5, type=float,
                    help='margin')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='alpha parameter for weighing losses')
parser.add_argument('--angle_rot', default=15, type=int, help='angle for rotation')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--tensorboard', default=True, type=str2bool, help='Use tensor board x for loss visualization')
parser.add_argument('--port', default=6006, type=int, help='set visdom port')
parser.add_argument('--send_images', type=str2bool, default=True,
                    help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_base_path', default=config['save_base_path'], help='Location to save all model data base path')
parser.add_argument('--mot_root', default=config['mot_root'], help='Location of MOT root directory')
parser.add_argument('--type', default=config['type'], help='train/test')
parser.add_argument('--show_image', default=True, help='show image if true, or hidden')
parser.add_argument('--save_video', default=True, help='save video if true')
parser.add_argument('--log_folder', help='video saving or result saving folder')
parser.add_argument('--mot_version', default=17, help='mot version')

args = parser.parse_args()

args.save_folder = args.save_base_path + 'weights'
args.save_images_folder = args.save_base_path + 'images'
