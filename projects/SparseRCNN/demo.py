import torch
from sparsercnn import SparseRCNN, add_sparsercnn_config

import cv2
import time
import urllib
import numpy as np
import sys
import torch.nn.functional as F

from sparsercnn.config import _C as cfg
yaml = 'projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml'
cfg.merge_from_file(yaml)

model = SparseRCNN(cfg).cuda()
model.eval()
# state_dict = torch.load('/workspace/mnt/storage/songqinglong/code/project/SparseR-CNN/output/model_final.pth')["model"]
# state_dict = torch.load('./output/model_final.pth.tar')
state_dict = torch.load('r50_100pro_3x_model.pth')

new_state_dict = {}
# for k, v in state_dict['state_dict'].items():
#     new_state_dict[k[7:]] = v
for k, v in state_dict.items():
    if ('conv' in k or 'fpn' in k or 'shortcut' in k) and 'norm' not in k:
        if 'weight' in k:
            new_state_dict[k.replace('weight', 'conv2d.weight')] = v
        if 'bias' in k:
            new_state_dict[k.replace('bias', 'conv2d.bias')] = v
    else:
        new_state_dict[k] = v
    
model.load_state_dict(new_state_dict)


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def prepare_data(img):
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(3, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    size_divisibility = 32

    img = [normalizer(_img) for _img in img]
    image_sizes = torch.Tensor([[im.shape[-2], im.shape[-1]] for im in img])
    max_size = image_sizes.max(0)[0].int().tolist()

    if size_divisibility > 1:
        stride = size_divisibility
        max_size = [(d + (stride - 1)) // stride * stride for d in max_size]

    if len(img) == 1:
        image_size = image_sizes[0].numpy().astype(np.int)
        padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
        batched_imgs = F.pad(img[0], padding_size, value=0.0).unsqueeze_(0)
    else:
        batch_shape = [len(img)] + list(img[0].shape[:-2]) + list(max_size)
        # print(batch_shape)
        batched_imgs = img[0].new_full(batch_shape, 0.0)
        for img, pad_img in zip(img, batched_imgs):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

    return batched_imgs.contiguous()

def main(image_root):
    print(image_root)
    # image_root = '/workspace/mnt/storage/songqinglong/song/data/车辆抓拍/金门20201115-20201122/0..jpg'

    image = cv2.imread(image_root)
    # image = url_to_image(image_root)
    image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    h, w, _ = image.shape
    # print(image.shape)
    dst_image = image.copy()
    image = image[:,:,(2,1,0)]
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image_tensor = prepare_data(image_tensor).cuda()
    image_whwh = torch.Tensor([[w, h, w, h]]).cuda()

    t0 = time.time()
    N = 1
    for i in range(N):
        scores, boxes, labels = model(image_tensor, image_whwh)
    t1 = time.time()

    for i in range(len(scores[0])):
        score = scores[0, i].cpu().data.numpy()
        label = labels[0, i].cpu().data.numpy()
        if score<0.4:
            continue
        print(score, label)
        box = boxes[0, i].cpu().data.numpy()
        cv2.rectangle(dst_image, (box[0],box[1]), (box[2],box[3]), (255,0,0))
    cv2.imwrite('result/'+ image_root.split('/')[-1], dst_image)
    print('done!')
    return 1000*(t1-t0)
if __name__ == '__main__':
    import glob
    for _r in glob.glob('../SparseR-CNN/images/*.jpg'):
        main(_r)
