import numpy as np
import cv2
import os, random
import torch
from torch.utils.data import Dataset
import json

# from utils.image import flip, color_aug
# from utils.image import get_affine_transform, affine_transform
# from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.data import transforms as T

import math

def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
  """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


class VCOCO(Dataset):
    # num_classes = 80
    # num_classes_verb = 25
    # default_resolution = [512, 512]
    # mean = np.array([0.40789654, 0.44719302, 0.47026115],
    #                 dtype=np.float32).reshape(1, 1, 3)
    # std = np.array([0.28863828, 0.27408164, 0.27809835],
    #                dtype=np.float32).reshape(1, 1, 3)
    def __init__(self, root_path, image_dir, split = 'train', resize_keep_ratio=False, multiscale_mode='value'):
        self.root = root_path
        self.image_dir = image_dir
        tfm_gens = []
        min_size = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
        max_size = 1333
        sample_style = 'choice'
        if split == 'train':
            tfm_gens.append(T.RandomFlip())
        tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
        self.tfm_gens = tfm_gens
        # self.crop_gen = [
        #         T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
        #         T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
        #     ]

        if split == 'train':
            self.hoi_annotations = json.load(open(os.path.join(self.root, 'annotations', 'train_vcoco.json'), 'r'))
            self.resize_keep_ratio = resize_keep_ratio
            self.multiscale_mode = multiscale_mode
            self.flip = True
            self.ids = []
            for i, vcoco in enumerate(self.hoi_annotations):
                flag_bad = 0
                for hoi in vcoco['hoi_annotation']:
                    if hoi['subject_id'] >= len(vcoco['annotations']) or hoi['object_id'] >= len(vcoco['annotations']):
                        flag_bad = 1
                        break
                if flag_bad == 0:
                    self.ids.append(i)
            if split == 'train':
                self.shuffle()
            self.num_classes = 80
            self.max_objs = 128
            self.max_rels = 64
            self._valid_ids = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                82, 84, 85, 86, 87, 88, 89, 90]
            self._valid_ids_verb = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 28]

            self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

            self.cat_ids_verb = {v: i for i, v in enumerate(self._valid_ids_verb)}
            
            self.split = split
            self.num_classes_verb = len(self._valid_ids_verb)

        else:
            self.hoi_annotations = json.load(open(os.path.join(self.root, 'annotations', 'test_vcoco.json'), 'r'))
            self.ids = list(range(len(self.hoi_annotations)))

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        batch_inputs = []
        
        img_id = self.ids[index]

        file_name = self.hoi_annotations[img_id]['file_name']
        img_path = os.path.join(self.root, self.image_dir, file_name)
        anns = self.hoi_annotations[img_id]['annotations']
        hoi_anns = self.hoi_annotations[img_id]['hoi_annotation']
        num_objs = min(len(anns), self.max_objs)
        img_path = img_path.replace('COCO_train2014_', '')
        img = cv2.imread(img_path)
        h, w , _ = img.shape

        # c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        # s = max(img.shape[0], img.shape[1]) * 1.0
        # input_h, input_w = 512, 512

        # flipped = False
        # if self.split == 'train':
        #     s = s * np.random.choice(np.arange(0.7, 1.4, 0.1))
        #     w_border = self._get_border(128, img.shape[1])
        #     h_border = self._get_border(128, img.shape[0])
        #     c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        #     c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

        #     if np.random.random() < 0.5:
        #         flipped = True
        #         img = img[:, ::-1, :]
        #         c[0] = w - c[0] - 1

        # trans_input = get_affine_transform(
        #     c, s, 0, [input_w, input_h])
        # img = cv2.warpAffine(img, trans_input,
        #                      (input_w, input_h),
        #                      flags=cv2.INTER_LINEAR)



        image_size = [h,w]*num_objs
        image_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        # image_tensor, transforms = T.apply_transform_gens(self.tfm_gens, img)
        
        num_rels = min(len(hoi_anns), self.max_rels)
        hoi_rel = []
        for k in range(num_rels):
            hoi = hoi_anns[k]
            sub_id = hoi['subject_id']
            obj_id = hoi['object_id']
            hoi_rel.append([sub_id, obj_id])
        hoi_rel = torch.Tensor(hoi_rel)

        # instances = Instances(image_size)
        gt_boxes = []
        gt_classes = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = np.asarray(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            gt_boxes.append(bbox)
            gt_classes.append(cls_id)
        gt_boxes = torch.Tensor(gt_boxes)
        gt_classes = torch.Tensor(gt_classes)
        # instances.gt_boxes = Boxes(gt_boxes)
        # instances.gt_classes = torch.Tensor(gt_classes)

        batch_inputs.append({'image': image_tensor, "gt_classes": gt_classes, "gt_boxes": gt_boxes, "hoi_rel": hoi_rel, 'image_size': torch.Tensor([h, w])})
        return batch_inputs

    def __len__(self):
        return len(self.ids)

    def shuffle(self):
        random.shuffle(self.ids)
