from torch.utils.data.sampler import Sampler
import random

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
        # if self.drop_last:
        #     l0 = len(self.data0) // self.batch_size
        #     l1 = len(self.data1) // self.batch_size
        # else:
        #     l0 = (len(self.data0) + self.batch_size - 1) // self.batch_size
        #     l1 = (len(self.data1) + self.batch_size - 1) // self.batch_size
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