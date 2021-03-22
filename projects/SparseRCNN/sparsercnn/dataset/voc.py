# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import xml.etree.ElementTree as ET
# from typing import List, Tuple, Union
import torch.utils.data as data
import cv2
import torch

# fmt: off
# class_names = (
#     "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#     "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
#     "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# )
class_names = (
    "vehicle",
)
# fmt: on

class VOCDataset(data.Dataset):
    def __init__(self, cfg, mode, transform=None):
        super(VOCDataset, self).__init__()
        assert mode in ['train', 'val']
        if mode == 'train':
            dataset = cfg.DATASETS.TRAIN[0]
        elif mode == 'val':
            dataset = cfg.DATASETS.TEST[0]
        self.is_rgb = cfg.INPUT.FORMAT
        
        root = cfg.BASE_ROOT
        self.transform = transform
        self.images_root = os.path.join(root, 'JPEGImages')
        self.annotations_root = os.path.join(root, 'Annotations')

        if mode == 'train':
            try:
                traintxt = open(os.path.join(root, 'ImageSets', 'Main', 'train.txt'), 'r')
            except:
                traintxt = open(os.path.join(root, 'ImageSets', 'train.txt'), 'r')
            self.data_list = []
            for _l in traintxt:
                self.data_list.append(_l.replace('\n', ''))
        
        else:
            try:
                valtxt = open(os.path.join(root, 'ImageSets', 'Main', 'val.txt'), 'r')
            except:
                valtxt = open(os.path.join(root, 'ImageSets', 'val.txt'), 'r')
            self.data_list = []
            for _l in valtxt:
                self.data_list.append(_l.replace('\n', ''))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        fileids = self.data_list[index]
        anno_file = os.path.join(self.annotations_root, fileids + ".xml")
        jpeg_file = os.path.join(self.images_root, fileids + ".jpg")

        tree = ET.parse(anno_file)

        gt_boxes = []
        gt_classes = []
        num_instances = 0

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            if cls == 'car' or cls == 'bus' or cls == 'vehicle':
                cls = 'vehicle'
            else:
                continue
            gt_boxes.append(bbox)
            gt_classes.append(class_names.index(cls))

        img = cv2.imread(jpeg_file)
        h, w, _ = img.shape
        if self.transform is not None:
            img, gt_boxes = self.transform(img, gt_boxes)

        if self.is_rgb == "RGB":
            try:
                img = img[:, :, [2,1,0]]
            except:
                print(fileids)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img_whwh = torch.as_tensor([img.shape[2], img.shape[1], img.shape[2], img.shape[1]])

        label = {'num_instances': num_instances,
                  'image_id': fileids,
                  'height': h,
                  'width': w,
                  'gt_boxes': torch.Tensor(gt_boxes),
                  'gt_classes': torch.LongTensor(gt_classes),
                  'image_size_xyxy': img_whwh,
                  'image_size_xyxy_tgt': img_whwh.unsqueeze(0).repeat(len(gt_boxes), 1)
                  }
        return img, img_whwh, label
