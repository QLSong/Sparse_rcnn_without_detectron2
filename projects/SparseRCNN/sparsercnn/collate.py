import torch
import torch.nn.functional as F
import numpy as np

class Collate(object):
    def __init__(self, cfg, size_divisibility=32):
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.size_divisibility = size_divisibility
        self.pad_value = 0
    
    def __call__(self, batch):
        img, img_whwh, target = zip(*batch)
        assert len(img) == len(img_whwh)
        assert len(img) == len(target)

        img = [self.normalizer(_img) for _img in img]

        image_sizes = torch.Tensor([[im.shape[-2], im.shape[-1]] for im in img])

        max_size = image_sizes.max(0)[0].int().tolist()

        if self.size_divisibility > 1:
            stride = self.size_divisibility
            max_size = [(d + (stride - 1)) // stride * stride for d in max_size]

        if len(img) == 1:
            image_size = image_sizes[0].numpy().astype(np.int)
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            batched_imgs = F.pad(img[0], padding_size, value=self.pad_value).unsqueeze_(0)
        else:
            batch_shape = [len(img)] + list(img[0].shape[:-2]) + list(max_size)
            batched_imgs = img[0].new_full(batch_shape, 0.0)
            for img, pad_img in zip(img, batched_imgs):
                pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

        img_whwh = torch.stack(img_whwh)

        return batched_imgs.contiguous(), img_whwh, target
