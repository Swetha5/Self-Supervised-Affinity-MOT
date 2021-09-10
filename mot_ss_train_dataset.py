"""
@project: ss-mot
@author: swetha
"""

import os, os.path
import pandas as pd
import cv2
import numpy as np
from config.config import config
import sys
import torch
import torch.utils.data as data
from PIL import Image, ImageDraw, ImageFont
import random
from utils.augmentations import SSJAugmentation
from config.config import config
from utils.arg_parse import args
from utils.data_utils import rotate_im, get_corners, rotate_box, get_enclosing_box, clip_box
import imgaug.augmenters as iaa
import albumentations as A


class GTSingleParser:
    def __init__(self, folder,
                 detector=config['detector'],
                 min_confidence=None,
                 min_visibility=config['min_visibility'],
                 min_gap=config['min_gap_frame'],
                 max_gap=config['max_gap_frame']):
        self.min_gap = min_gap
        self.max_gap = max_gap

        # 1. get the det path and image folder
        det_file_path = os.path.join(folder, 'det/det.txt')
        self.folder = folder

        # 2. read detection data
        self.detection = pd.read_csv(det_file_path, header=None)
        if min_confidence is not None:
            self.detection = self.detection[self.detection[6] > min_confidence]
        self.detection_group = self.detection.groupby(0)
        self.detection_group_keys = list(self.detection_group.indices.keys())
        self.max_frame_index = max(self.detection_group_keys)
        self.recorder = {}
        self.blur_aug = iaa.Sequential([iaa.MotionBlur(k=10, angle=[-25, 25])])

        for key in self.detection_group_keys:
            det = self.detection_group.get_group(key).values
            det = np.array(det[:, 2:6])
            det[:, 2:4] += det[:, :2]

            self.recorder[key] = det

    def get_detection(self, frame_index):
        if frame_index > len(self.detection_group_keys) or self.detection_group_keys.count(frame_index) == 0:
            return None
        return self.recorder[frame_index]

    def get_image(self, frame_index):
        if frame_index > len(self.detection_group_keys):
            return None
        image_path = os.path.join(self.folder, 'img1/{0:06}.jpg'.format(frame_index))
        return cv2.imread(image_path)

    def get_num_objects(self, frame_index):
        return len(self.recorder[frame_index])

    def mirror(self, image, boxes):
        _, width, _ = image.shape
        image = np.array(image[:, ::-1])
        boxes = boxes.copy()
        boxes[:, 0] = width - boxes[:, 0]
        boxes[:, 2] = width - boxes[:, 2]
        return image, boxes

    def rotation(self, image, boxes, angle):
        w, h = image.shape[1], image.shape[0]
        cx, cy = w // 2, h // 2

        img = rotate_im(image, angle)

        corners = get_corners(boxes)
        corners = np.hstack((corners, boxes[:, 4:]))
        corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

        new_bbox = get_enclosing_box(corners)

        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h
        img = cv2.resize(img, (w, h))

        new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
        bboxes = new_bbox
        bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)

        return img, bboxes

    def random_blur(self, image, blur_prob=0.5):
        ssaug_transform = A.Compose([A.MotionBlur(blur_limit=(5, 21), p=blur_prob)
                                     ])
        transformed_image = ssaug_transform(image=image)["image"]
        return transformed_image

    def random_cutout(self, image, prob=0.5):
        ct_transform = A.Compose([A.CoarseDropout(max_holes=50, max_height=45, max_width=45, min_holes=35,
                                                  min_height=20, min_width=20, fill_value=0, p=prob)])
        transformed_image = ct_transform(image=image)["image"]
        return transformed_image

    def get_item(self, frame_index):
        """
        get the current_image, current_boxes, next_image, next_boxes, labels from the frame_index
        if args.cyc_sampling then sample frames and boxes to be trained with asCyc loss
        :param frame_index:
        :return: current_image, current_boxes, next_image, next_boxes, labels
        """

        orig_img = self.get_image(frame_index)
        orig_boxes = self.get_detection(frame_index)

        if orig_img is None or orig_boxes is None:
            return None, None, None, None, None

        if args.cyc_sampling:
            if frame_index + self.max_gap <= self.max_frame_index:
                next_frame_index = frame_index + random.randint(self.min_gap, self.max_gap)
                gap_range = self.max_gap
                gap_min = self.min_gap
            else:
                gap_range = self.max_frame_index - frame_index
                gap_min = 0
                next_frame_index = frame_index + random.randint(gap_min, gap_range)

            while not next_frame_index in self.recorder:
                next_frame_index = frame_index + random.randint(gap_min, gap_range)

            next_img = self.get_image(next_frame_index)
            next_boxes = self.get_detection(next_frame_index)
            if next_img is None:
                return None, None, None, None, None

            num_objects = self.get_num_objects(frame_index)

            #placeholder
            labels = np.repeat(np.expand_dims(np.array(np.zeros(num_objects)), axis=1), num_objects,
                               axis=1) == np.repeat(np.expand_dims(np.zeros(num_objects), axis=0),
                                                    num_objects, axis=0)

            return orig_img, np.array(orig_boxes), next_img, np.array(next_boxes), labels
        else:
            aug_choice = random.choice(['rot', 'flip', 'blur', 'cutout'])
            print('aug choice is :' + aug_choice)
            if aug_choice == 'flip':
                aug_img, aug_boxes = self.mirror(orig_img, orig_boxes)
                # randomly apply motion blur
                aug_img = self.random_blur(aug_img, 0.5)

                num_objects = self.get_num_objects(frame_index)
                obj_indices = np.arange(num_objects)

                labels = np.repeat(np.expand_dims(np.array(obj_indices), axis=1), num_objects,
                                   axis=1) == np.repeat(np.expand_dims(np.array(obj_indices), axis=0),
                                                        len(obj_indices), axis=0)
                return orig_img, np.array(orig_boxes), aug_img, np.array(aug_boxes), labels
            elif aug_choice == 'rot':
                aug_img, aug_boxes = self.rotation(orig_img, orig_boxes, random.choice(
                    np.arange(-args.angle_rot, args.angle_rot)))
                aug_img = self.random_blur(aug_img, 0.5)

                num_objects = self.get_num_objects(frame_index)
                obj_indices = np.arange(num_objects)

                labels = np.repeat(np.expand_dims(np.array(obj_indices), axis=1), num_objects,
                                   axis=1) == np.repeat(np.expand_dims(np.array(obj_indices), axis=0),
                                                        len(obj_indices), axis=0)

                return orig_img, np.array(orig_boxes), aug_img, np.array(aug_boxes), labels
            elif aug_choice in ['blur', 'cutout']:
                if aug_choice == 'blur':
                    aug_img = self.random_blur(orig_img, 1.0)
                else:
                    aug_img = self.random_cutout(orig_img, 1.0)
                num_objects = self.get_num_objects(frame_index)
                obj_indices = np.arange(num_objects)
                labels = np.repeat(np.expand_dims(np.array(obj_indices), axis=1), num_objects,
                                   axis=1) == np.repeat(np.expand_dims(np.array(obj_indices), axis=0),
                                                        len(obj_indices), axis=0)

                return orig_img, np.array(orig_boxes), aug_img, np.array(orig_boxes), labels

    def __len__(self):
        return self.max_frame_index


class GTParser:
    def __init__(self, mot_root=config['mot_root'],
                 detector=config['detector'],
                 type=config['type']):
        # 1. get all the folders
        mot_root = os.path.join(mot_root, type)
        all_folders = sorted(
            [os.path.join(mot_root, i) for i in os.listdir(mot_root)
             if os.path.isdir(os.path.join(mot_root, i))
             and i.find(detector) != -1]
        )
        # 2. create single parser
        self.parsers = [GTSingleParser(folder) for folder in all_folders]

        # 3. get some basic information
        self.lens = [len(p) for p in self.parsers]
        self.len = sum(self.lens)

    def __len__(self):
        # get the length of all the matching frame
        return self.len

    def __getitem__(self, item):
        if item < 0:
            return None, None, None, None, None
        # 1. find the parser
        total_len = 0
        index = 0
        current_item = item
        for l in self.lens:
            total_len += l
            if item < total_len:
                break
            else:
                index += 1
                current_item -= l

        # 2. get items
        if index >= len(self.parsers):
            return None, None, None, None, None
        return self.parsers[index].get_item(current_item)


class MOTSSTrainDataset(data.Dataset):
    """
    The class is the dataset for train, which read gt.txt file and rearrange them as the tracks set.
    it can be selected from the specified frame
    """

    def __init__(self,
                 mot_root=config['mot_root'],
                 transform=None,
                 type=config['type'],
                 detector=config['detector'],
                 max_object=config['max_object'],
                 dataset_name='MOT17'
                 ):
        # 1. init all the variables
        self.mot_root = mot_root
        self.transform = transform
        self.type = type
        self.detector = detector
        self.max_object = max_object
        self.dataset_name = dataset_name

        # 2. init GTParser
        self.parser = GTParser(self.mot_root, self.detector)
        print('Performing cyc sampling ? ' + str(args.cyc_sampling))

    def __getitem__(self, item):
        current_image, current_box, next_image, next_box, labels = self.parser[item]

        while current_image is None:
            current_image, current_box, next_image, next_box, labels = self.parser[
                item + random.randint(-config['max_gap_frame'], config['max_gap_frame'])]
            print('None processing.')

        if self.transform is None:
            return current_image, next_image, current_box, next_box, labels

        # change the label to max_object x max_object
        labels = np.pad(labels,
                        [(0, self.max_object - labels.shape[0]),
                         (0, self.max_object - labels.shape[1])],
                        mode='constant',
                        constant_values=0)
        return self.transform(current_image, next_image, current_box, next_box, labels)

    def __len__(self):
        return len(self.parser)


class MOTDataReader:
    def __init__(self, image_folder, detection_file_name, min_confidence=None):
        self.image_folder = image_folder
        self.detection_file_name = detection_file_name
        self.image_format = os.path.join(self.image_folder, '{0:06d}.jpg')
        self.detection = pd.read_csv(self.detection_file_name, header=None)
        if min_confidence is not None:
            self.detection = self.detection[self.detection[6] > min_confidence]
        self.detection_group = self.detection.groupby(0)
        self.detection_group_keys = list(self.detection_group.indices.keys())

    def __len__(self):
        return len(self.detection_group_keys)

    def get_detection_by_index(self, index):
        if index > len(self.detection_group_keys) or self.detection_group_keys.count(index) == 0:
            return None
        return self.detection_group.get_group(index).values

    def get_image_by_index(self, index):
        if index > len(self.detection_group_keys):
            return None

        return cv2.imread(self.image_format.format(index))

    def __getitem__(self, item):
        return (self.get_image_by_index(item + 1),
                self.get_detection_by_index(item + 1))
