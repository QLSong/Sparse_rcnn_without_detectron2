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
# CLASS_NAMES = (
#     "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#     "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
#     "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# )
class_names = (
    "vehicle",
)
# fmt: on

class VOCDataset(data.Dataset):
    def __init__(self, root, transform=None, mode='train'):
        super(VOCDataset, self).__init__()
        self.transform = transform
        self.images_root = os.path.join(root, 'JPEGImages')
        self.annotations_root = os.path.join(root, 'Annotations')

        if mode == 'train':
            traintxt = open(os.path.join(root, 'ImageSets', 'train.txt'), 'r')
            self.data_list = []
            for _l in traintxt:
                self.data_list.append(_l.replace('\n', ''))
        
        else:
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

        # with PathManager.open(anno_file) as f:
        tree = ET.parse(anno_file)

        gt_boxes = []
        gt_classes = []
        num_instances = 0

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            if cls == 'car' or cls == 'bus' or cls == 'vehicle':
                cls = 'vehicle'
            else:
                continue
            gt_boxes.append(bbox)
            gt_classes.append(class_names.index(cls))

        img = cv2.imread(jpeg_file)
        h, w, _ = img.shape
        
        if self.transform is not None:
            img = self.transform(img)
        try:
            img = img[:, :, [2,1,0]]
        except:
            print(fileids)
        img = img.transpose(2, 0, 1)
        img = torch.Tensor(img)

        label = {'num_instances': num_instances,
                  'height': h,
                  'width': w,
                  'gt_boxes': gt_boxes,
                  'gt_classes': gt_classes
                  }
        img_whwh = torch.tensor([w, h, w, h])
        return img, img_whwh, label
