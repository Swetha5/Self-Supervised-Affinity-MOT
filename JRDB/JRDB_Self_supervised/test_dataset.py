import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from functools import partial
import torch
from test_dataset_preprocess import get_pointcloud,create_frame_detections,get_frame_det_info,get_test_names

class TestSequenceDataset(object):
    def __init__(self, root_dir, label_dir, fix_iou=0.2, fix_count=2, gt_ratio=0, sample_max_len=2):
        
        self.root_dir = root_dir
        self.sample_max_len = sample_max_len
        self.label_dir = label_dir
        #self.test = False
        #self.val_seqs=['clark-center-2019-02-28_1']
        #self.val_seqs=['bytes-cafe-2019-02-07_0','clark-center-2019-02-28_1']
        self.val_seqs=get_test_names()
        #self.val_seqs=get_train_names()
        self.get_pointcloud = get_pointcloud
        self.metas = self._generate_meta_seq()

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        return self.metas[idx]
    
    def _generate_meta_seq(self):
        #root_dir='train_dataset/pointclouds_pt/'
        metas = []
        for seq_id in self.val_seqs:
            name=seq_id
            #fname='/home/c3-0/datasets/JRDB/cvgl/group/jrdb/data/test_dataset/detections/detections_3d/'+seq_id+'.json'
            fname=self.label_dir+seq_id+'.json'
            
            #fname='/home/c3-0/datasets/JRDB/cvgl/group/jrdb/data/train_dataset/labels/labels_3d/'+seq_id+'.json'
            with open(fname) as json_file:
                orig_labels = json.load(json_file)

            labels=orig_labels['detections']
            frame_id=list(sorted(orig_labels['detections'].keys()))

            #labels=orig_labels['labels']
            #frame_id=list(sorted(orig_labels['labels'].keys()))

            det_seq=[]
            for i in range(0, len(frame_id)-1):
                #Each test sample must contains two frames
                det_frames=[]

                #frame_1
                frame_1={}
                frame_1['seq_id']=seq_id
                frame_1['frame_id']=frame_id[i]
                frame_1['objects']=labels[frame_id[i]]
                frame_1['detection']=create_frame_detections(frame_1)
                det_frames.append(frame_1)


                #frame_2
                frame_2={}
                frame_2['seq_id']=seq_id
                frame_2['frame_id']=frame_id[i+1]
                frame_2['objects']=labels[frame_id[i+1]]
                frame_2['detection']=create_frame_detections(frame_2)
                det_frames.append(frame_2)

                #test sample of combined frames
                det_seq.append(det_frames)

            metas.append(TestSequence(name, self.root_dir,det_seq, self.get_pointcloud,interval=2))
        return metas    


class TestSequence(Dataset):
    def __init__(self,name, root_dir, det_frames, get_pointcloud_partial, interval=2):
        self.det_frames = det_frames
        self.interval = interval
        self.root_dir = root_dir
        self.metass= det_frames
        self.idx = 0
        self.seq_len = len(det_frames) + interval -1
        self.name = name
        self.get_pointcloud = get_pointcloud_partial
        '''

        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
        assert len(det_frames) == len(det_frames)
        '''

    def __getitem__(self, idx):
        return self._generate_lidar(idx)

    def __len__(self):
        return len(self.metass)
    
    def _generate_lidar(self, idx):
        frames = self.metass[idx]
        det_imgs = []
        det_split = []
        dets = []
        det_info = get_frame_det_info()
        first_flag = 0
        for frame in frames:
            seq_id=frame['seq_id']
            frame_id=frame['frame_id'].split('.')[0]

            point_cloud=get_pointcloud(frame['detection'],seq_id,frame_id,self.root_dir)


            det_num = frame['detection']['bbox'].shape[0]
            det_num=min(det_num,80)
            dets.append(frame['detection'])

            det_split.append(det_num)
            det_info['loc'].append(torch.Tensor(frame['detection']['location']))
            det_info['rot'].append(torch.Tensor(frame['detection']['rotation_y']))
            det_info['dim'].append(torch.Tensor(frame['detection']['dimensions']))
            det_info['points'].append(torch.Tensor(point_cloud['points']))
            det_info['points_split'].append(torch.Tensor(point_cloud['points_split'])[first_flag:])
            det_info['info_id'].append(seq_id+'/'+frame_id)

            if first_flag == 0:
                first_flag += 1

        det_imgs = []
        det_info['loc'] = torch.cat(det_info['loc'], dim=0)
        det_info['rot'] = torch.cat(det_info['rot'], dim=0)
        det_info['dim'] = torch.cat(det_info['dim'], dim=0)
        det_info['points'] = torch.cat(det_info['points'], dim=0)
        start = 0
        for i in range(len(det_info['points_split'])):
            det_info['points_split'][i] += start
            start = det_info['points_split'][i][-1]
        det_info['points_split'] = torch.cat(det_info['points_split'], dim=0)
        return det_imgs, det_info, dets, det_split

