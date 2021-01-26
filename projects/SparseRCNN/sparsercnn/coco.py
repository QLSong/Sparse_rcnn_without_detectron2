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

from pycocotools.cocoeval import COCOeval
# from detectron2.structures import ImageList
# from utils import zipreader

# logger = logging.getLogger(__name__)


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

    def __init__(self, root, dataset, transform=None,
                 target_transform=None):
        from pycocotools.coco import COCO
        self.name = 'COCO'
        self.root = root
        self.dataset = dataset
        # print(self._get_anno_file_name())
        self.coco = COCO(self._get_anno_file_name())
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        # logger.info('=> classes: {}'.format(self.classes))
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

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            img = img[:, :, [2,1,0]]
        except:
            print(self._get_image_path(file_name).replace('images/', ''))
        h, w, _ = img.shape
        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        img = self.preprocess_image(img)
        label = {'num_instances': len(target),
                  'height': h,
                  'width': w,
                  'gt_boxes': [t['bbox'] for t in target],
                  'gt_classes': [t['category_id'] for t in target]
                  }
        img_whwh = torch.tensor([w, h, w, h])
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

    def preprocess_image(self, images):
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images

if __name__ == '__main__':
    dataset = CocoDataset('datasets/coco', 'train2017')
    print(dataset[0][0].shape)
    print(dataset[0][1])
