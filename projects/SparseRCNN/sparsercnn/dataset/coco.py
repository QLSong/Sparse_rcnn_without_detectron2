# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import os.path

import cv2
import json
import numpy as np
from torch.utils.data import Dataset
import torch

# coco_id_name_map={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
#                    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
#                    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
#                    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
#                    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
#                    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
#                    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
#                    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
#                    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
#                    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
#                    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
#                    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
#                    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
#                    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
#                    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
#                    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
coco_id_idx_map=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 
                15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
                27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
                40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 
                63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 
                78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

from pycocotools.cocoeval import COCOeval

class CocoDataset(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where dataset is located to.
        dataset (string): Dataset name(train2017, val2017, test2017).
        data_format(string): Data format for reading('jpg', 'zip')
        transform (callable, optional): A function/transform that  takes in an opencv image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, cfg, mode, transform=None):
        from pycocotools.coco import COCO
        super(CocoDataset, self).__init__()
        assert mode in ['train', 'val']
        if mode == 'train':
            dataset = cfg.DATASETS.TRAIN[0]
        elif mode == 'val':
            dataset = cfg.DATASETS.TEST[0]
        self.is_rgb = cfg.INPUT.FORMAT
        self.name = 'COCO'
        self.root = './datasets/coco'
        self.mode = dataset.split('_')[-1]
        self.dataset = dataset.split('_')[-1]+dataset.split('_')[-2]
        self.coco = COCO(self._get_anno_file_name())
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

    def _get_anno_file_name(self):
        # example: root/annotations/person_keypoints_tran2017.json
        # image_info_test-dev2017.json
        if 'test' in self.dataset:
            return os.path.join(
                self.root,
                'annotations',
                'image_info_{}.json'.format(
                    self.dataset
                )
            )
        else:
            return os.path.join(
                self.root,
                'annotations',
                'instances_{}.json'.format(
                    self.dataset
                )
            )

    def _get_image_path(self, file_name):
        images_dir = os.path.join(self.root, 'images')
        dataset = 'test2017' if 'test' in self.dataset else self.dataset
        return os.path.join(images_dir, dataset, file_name)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        file_name = coco.loadImgs(img_id)[0]['file_name']

        img = cv2.imread(
            self._get_image_path(file_name).replace('images/', ''),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        h, w, _ = img.shape
        if self.transform is not None:
            img = self.transform(img)
        if self.is_rgb == "RGB":
            try:
                img = img[:, :, [2,1,0]]
            except:
                print(self._get_image_path(file_name).replace('images/', ''))
        img = img.transpose(2, 0, 1)
        img = torch.Tensor(img)
        img_whwh = torch.Tensor([img.shape[2], img.shape[1], img.shape[2], img.shape[1]])
        # print(target)
        gt_boxes = [[t['bbox'][0], t['bbox'][1], t['bbox'][0]+t['bbox'][2], t['bbox'][1]+t['bbox'][3]] for t in target]
        gt_classes = [coco_id_idx_map.index(t['category_id']) for t in target]
        if len(gt_classes) == 0:
            return self[index+1]

        label = {'num_instances': len(gt_classes),
                  'image_id': img_id,
                  'height': h,
                  'width': w,
                  'gt_boxes': torch.Tensor(gt_boxes)/torch.Tensor([w, h, w, h])*img_whwh,
                  'gt_classes': torch.LongTensor(gt_classes),
                  'image_size_xyxy': img_whwh,
                  'image_size_xyxy_tgt': img_whwh.unsqueeze(0).repeat(len(gt_boxes), 1)
                  }
        
        return img, img_whwh, label

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

if __name__ == '__main__':
    dataset = CocoDataset('datasets/coco', 'train2017')
    print(dataset[0][0].shape)
    print(dataset[0][1])
