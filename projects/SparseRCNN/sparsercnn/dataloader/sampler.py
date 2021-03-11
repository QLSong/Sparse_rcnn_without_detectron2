from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
import random
import torch.distributed as dist
import torch
import math
import numpy as np

class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        # _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = num_replicas
        if rank is None:
            rank = rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0
        # assert hasattr(self.dataset, 'flag')
        self.flag = np.zeros(len(self.dataset), dtype=np.uint8)
        for i in range(len(self.dataset)):
            if self.dataset.image_aspect_ratio(i)>1:
                self.flag[i] = 1
            # img_info = self.data_infos[i]
            # if img_info['width'] / img_info['height'] > 1:
            #     self.flag[i] = 1
        # self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                indice = indice[list(
                    torch.randperm(int(size), generator=g).numpy())].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        return self.l0 + self.l1

    def group_images(self):
        data0 = []
        data1 = []
        for i in range(len(self.data_source)):
            if self.data_source.image_aspect_ratio(i) > 1.0:
                data0.append(i)
            else:
                data1.append(i)
        random.shuffle(data0)
        random.shuffle(data1)
        self.data0 = data0
        self.data1 = data1
        if self.drop_last:
            self.l0 = len(self.data0) // self.batch_size
            self.l1 = len(self.data1) // self.batch_size
        else:
            self.l0 = (len(self.data0) + self.batch_size - 1) // self.batch_size
            self.l1 = (len(self.data1) + self.batch_size - 1) // self.batch_size
        return [self.data0[i:i+self.batch_size] for i in range(0, self.l0*self.batch_size, self.batch_size)] \
                + [self.data1[i:i+self.batch_size] for i in range(0, self.l1*self.batch_size, self.batch_size)]

# class AspectRatioBasedSampler(Sampler):

#     def __init__(self, data_source, batch_size, drop_last):
#         self.data_source = data_source
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         self.groups = self.group_images()

#     def __iter__(self):
#         random.shuffle(self.groups)
#         for group in self.groups:
#             yield group

#     def __len__(self):
#         if self.drop_last:
#             return len(self.data_source) // self.batch_size
#         else:
#             return (len(self.data_source) + self.batch_size - 1) // self.batch_size

#     def group_images(self):
#         # determine the order of the images
#         order = list(range(len(self.data_source)))
#         order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

#         # divide into groups, one group = one batch
#         return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
        
class RandomSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        # if self.drop_last:
        #     return len(self.data_source) // self.batch_size
        # else:
        #     return (len(self.data_source) + self.batch_size - 1) // self.batch_size
        return self.l

    def group_images(self):
        order = list(range(len(self.data_source)))
        random.shuffle(order)
        if self.drop_last:
            self.l = len(self.data_source) // self.batch_size
        else:
            self.l = (len(self.data_source) + self.batch_size - 1) // self.batch_size
        return [order[i:i+self.batch_size] for i in range(0, self.l*self.batch_size, self.batch_size)]

