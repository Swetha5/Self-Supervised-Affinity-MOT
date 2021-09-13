"""
SST tracker net

Thanks to ssd pytorch implementation (see https://github.com/amdegroot/ssd.pytorch)
copyright: shijie Sun (shijieSun@chd.edu.cn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from config.config import config
import numpy as np
import os
from pointnet import *

#todo: add more extra columns to use the mogranic method
#todo: label the mot17 data and train the detector.
#todo: Does the inherient really work
#todo: add achors to extract features
#todo: think about how to represent the motion model
#todo: debug every feature step and see the feature change of each objects [do]
#todo: change the output of the SST.
#todo: add relu to extra net





class SST(nn.Module):
    #new: combine two vgg_net
    def __init__(self, phase, pointnet, final_net,cuda_device):
        super(SST, self).__init__()
        self.phase = phase


        self.stacker2_bn = nn.BatchNorm2d(512)
        self.final_dp = nn.Dropout(0.7)
        self.final_net = nn.ModuleList(final_net)
        
        self.pointnet=pointnet
        self.false_objects_column = None
        self.false_objects_row = None
        self.false_constant = 10
        #config['false_constant'] 
        self.use_gpu = cuda_device
        self.max_object=80
        #self.pointnet=pointnet

    def forward(self, input, det_info, det_id, det_cls, det_split):
        
        if self.use_gpu:
            pointnet=self.pointnet.cuda()
            points, trans = pointnet(
                Variable(det_info['points'].transpose(-1, -2).cuda()),
                Variable(det_info['points_split'].long().squeeze(0).cuda()))
        else:
            pointnet=self.pointnet
            
            points, trans = pointnet(
                Variable(det_info['points'].transpose(-1, -2)),
                Variable(det_info['points_split'].long().squeeze(0)))
        
        # x_pre.register_hook(lambda grad: print('selector_stacker1:', grad.sum().data[0]))
        # [B, N, N, C]
        x_pre=points[:det_split[0]]
        x_next=points[det_split[0]:]
        
        #Reshape x_pre
        m = nn.ConstantPad2d((0, 0, 0, 80-x_pre.shape[0]), 0)
        x_pre=m(x_pre)
        x_pre=torch.unsqueeze(x_pre, 0)
        
        #Reshape x_next
        m = nn.ConstantPad2d((0, 0, 0, 80-x_next.shape[0]), 0)
        x_next=m(x_next)
        x_next=torch.unsqueeze(x_next, 0)
        
        #print('x pre shape',x_pre.shape)
        #print('x next shape',x_next.shape)
        
        x = self.forward_stacker2(
            x_pre, x_next
        )
        
        x = self.final_dp(x)
        # [B, N, N, 1]
        #print('x_shape forward final',x.contiguous().shape)
        #print('final_next',self.final_net)
        x = self.forward_final(x, self.final_net)

        # add false unmatched row and column
        x = self.add_unmatched_dim(x)
        return x

    def forward_feature_extracter(self, x, l):
        '''
        extract features from the vgg layers and extra net
        :param x:
        :param l:
        :return: the features
        '''
        s = list()

        x = self.forward_vgg(x, self.vgg, s)
        x = self.forward_extras(x, self.extras, s)
        x = self.forward_selector_stacker1(s, l, self.selector)

        return x

    def get_similarity(self, image1, detection1, image2, detection2):
        feature1 = self.forward_feature_extracter(image1, detection1)
        feature2 = self.forward_feature_extracter(image2, detection2)
        return self.forward_stacker_features(feature1, feature2, False)


    def resize_dim(self, x, added_size, dim=1, constant=0):
        if added_size <= 0:
            return x
        shape = list(x.shape)
        shape[dim] = added_size
        if self.use_gpu:
            new_data = Variable(torch.ones(shape)*constant).cuda()
        else:
            new_data = Variable(torch.ones(shape) * constant)
        return torch.cat([x, new_data], dim=dim)

    def forward_stacker_features(self, xp, xn, fill_up_column=True):
        pre_rest_num = self.max_object - xp.shape[1]
        next_rest_num = self.max_object - xn.shape[1]
        pre_num = xp.shape[1]
        next_num = xn.shape[1]
        x = self.forward_stacker2(
            self.resize_dim(xp, pre_rest_num, dim=1),
            self.resize_dim(xn, next_rest_num, dim=1)
        )

        x = self.final_dp(x)
        # [B, N, N, 1]
        x = self.forward_final(x, self.final_net)
        x = x.contiguous()
        # add zero
        if next_num < self.max_object:
            x[0, 0, :, next_num:] = 0
        if pre_num < self.max_object:
            x[0, 0, pre_num:, :] = 0
        x = x[0, 0, :]
        # add false unmatched row and column
        x = self.resize_dim(x, 1, dim=0, constant=self.false_constant)
        x = self.resize_dim(x, 1, dim=1, constant=self.false_constant)

        x_f = F.softmax(x, dim=1)
        x_t = F.softmax(x, dim=0)
        # slice
        last_row, last_col = x_f.shape
        row_slice = list(range(pre_num)) + [last_row-1]
        col_slice = list(range(next_num)) + [last_col-1]
        x_f = x_f[row_slice, :]
        x_f = x_f[:, col_slice]
        x_t = x_t[row_slice, :]
        x_t = x_t[:, col_slice]

        x = Variable(torch.zeros(pre_num, next_num+1))
        # x[0:pre_num, 0:next_num] = torch.max(x_f[0:pre_num, 0:next_num], x_t[0:pre_num, 0:next_num])
        x[0:pre_num, 0:next_num] = (x_f[0:pre_num, 0:next_num] + x_t[0:pre_num, 0:next_num]) / 2.0
        x[:, next_num:next_num+1] = x_f[:pre_num, next_num:next_num+1]
        if fill_up_column and pre_num > 1:
            x = torch.cat([x, x[:, next_num:next_num+1].repeat(1, pre_num-1)], dim=1)

        if self.use_gpu:
            y = x.data.cpu().numpy()
            # del x, x_f, x_t
            # torch.cuda.empty_cache()
        else:
            y = x.data.numpy()

        return y

    def forward_selector_stacker1(self, sources, labels, selector):
        '''
        :param sources: [B, C, H, W]
        :param labels: [B, N, 1, 1, 2]
        :return: the connected feature
        '''
        sources = [
            F.relu(net(x), inplace=True) for net, x in zip(selector, sources)
        ]

        res = list()
        for label_index in range(labels.size(1)):
            label_res = list()
            for source_index in range(len(sources)):
                # [N, B, C, 1, 1]
                label_res.append(
                    # [B, C, 1, 1]
                    F.grid_sample(sources[source_index],  # [B, C, H, W]
                                  labels[:, label_index, :]  # [B, 1, 1, 2
                                  ).squeeze(2).squeeze(2)
                )
            res.append(torch.cat(label_res, 1))

        return torch.stack(res, 1)

    def forward_stacker2(self, stacker1_pre_output, stacker1_next_output):
        stacker1_pre_output = stacker1_pre_output.unsqueeze(2).repeat(1, 1, self.max_object, 1).permute(0, 3, 1, 2)
        stacker1_next_output = stacker1_next_output.unsqueeze(1).repeat(1, self.max_object, 1, 1).permute(0, 3, 1, 2)

        stacker1_pre_output = self.stacker2_bn(stacker1_pre_output.contiguous())
        stacker1_next_output = self.stacker2_bn(stacker1_next_output.contiguous())

        output = torch.cat(
            [stacker1_pre_output, stacker1_next_output],
            1
        )

        return output

    def forward_final(self, x, final_net):
        x = x.contiguous()
        for f in final_net:
            x = f(x)
        return x

    def add_unmatched_dim(self, x):
        if self.false_objects_column is None:
            self.false_objects_column = Variable(torch.ones(x.shape[0], x.shape[1], x.shape[2], 1)) * self.false_constant
            if self.use_gpu:
                self.false_objects_column = self.false_objects_column.cuda()
        x = torch.cat([x, self.false_objects_column], 3)

        if self.false_objects_row is None:
            self.false_objects_row = Variable(torch.ones(x.shape[0], x.shape[1], 1, x.shape[3])) * self.false_constant
            if self.use_gpu:
                self.false_objects_row = self.false_objects_row.cuda()
        x = torch.cat([x, self.false_objects_row], 2)
        return x

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(
                torch.load(base_file,
                           map_location=lambda storage, loc: storage)
            )
            print('Finished')
        else:
            print('Sorry only .pth and .pkl files supported.')


def add_final(cfg, batch_normal=True):
    layers = []
    in_channels = int(cfg[0])
    layers += []
    # 1. add the 1:-2 layer with BatchNorm
    for v in cfg[1:-2]:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)
        if batch_normal:
            layers += [conv2d, nn.GroupNorm(v,v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    # 2. add the -2: layer without BatchNorm for BatchNorm would make the output value normal distribution.
    for v in cfg[-2:]:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v

    return layers


pointnet=PointNet_v1(3,512)
def build_model(phase,cuda_device):
    '''
    create the SSJ Tracker Object
    :return: ssj tracker object
    '''
    if phase != 'test' and phase != 'train':
        print('Error: Phase not recognized')
        return
    #base = config['base_net']
    #extras = config['extra_net']
    #final = config['final_net']


    return SST(phase,
               pointnet,
               add_final([1024, 512, 256, 128, 64, 1]),
               cuda_device
               )
