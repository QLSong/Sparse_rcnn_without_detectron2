import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import math

class Conv_bn_relu(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 kernel_size=3,
                 stride=1,
                 pad=1,
                 use_relu=True,
                 relu_name='ReLU'):
        super(Conv_bn_relu, self).__init__()
        self.use_relu = use_relu
        if self.use_relu and relu_name == 'ReLU':
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=False),
            )
        elif self.use_relu and relu_name == 'LeakyReLU':
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
                nn.LeakyReLU(inplace=False),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.convs(x)
        return out

class StemBlock(nn.Module):
    def __init__(self, inp=3, num_init_features=32):
        super(StemBlock, self).__init__()
        self.stem_1 = Conv_bn_relu(inp, num_init_features, 3, 2, 1)
        self.stem_2a = Conv_bn_relu(num_init_features,
                                    int(num_init_features / 2), 1, 1, 0)
        self.stem_2b = Conv_bn_relu(int(num_init_features / 2),
                                    num_init_features, 3, 2, 1)
        self.stem_2p = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem_3 = Conv_bn_relu(num_init_features * 2, num_init_features, 1,
                                   1, 0)

    def forward(self, x):
        stem_1_out = self.stem_1(x)
        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)
        stem_2p_out = self.stem_2p(stem_1_out)
        out = self.stem_3(torch.cat((stem_2b_out, stem_2p_out), 1))
        return out


class DenseBlock(nn.Module):
    def __init__(self, inp, inter_channel, growth_rate):
        super(DenseBlock, self).__init__()
        self.cb1_a = Conv_bn_relu(inp, inter_channel, 1, 1, 0)
        self.cb1_b = Conv_bn_relu(inter_channel, growth_rate, 3, 1, 1)
        self.cb2_a = Conv_bn_relu(inp, inter_channel, 1, 1, 0)
        self.cb2_b = Conv_bn_relu(inter_channel, growth_rate, 3, 1, 1)
        self.cb2_c = Conv_bn_relu(growth_rate, growth_rate, 3, 1, 1)

    def forward(self, x):
        cb1_a_out = self.cb1_a(x)
        cb1_b_out = self.cb1_b(cb1_a_out)
        cb2_a_out = self.cb2_a(x)
        cb2_b_out = self.cb2_b(cb2_a_out)
        cb2_c_out = self.cb2_c(cb2_b_out)
        out = torch.cat((x, cb1_b_out, cb2_c_out), 1)
        return out


class TransitionBlock(nn.Module):
    def __init__(self, inp, oup, with_pooling=True):
        super(TransitionBlock, self).__init__()
        if with_pooling:
            self.tb = nn.Sequential(Conv_bn_relu(inp, oup, 1, 1, 0),
                                    nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.tb = Conv_bn_relu(inp, oup, 1, 1, 0)

    def forward(self, x):
        out = self.tb(x)
        return out

class PeleeNet(nn.Module):
    def __init__(self,
                #  cfg,
                 num_classes=1000,
                 num_init_features=32,
                 growthRate=32,
                 nDenseBlocks=[3, 4, 8, 6],
                 bottleneck_width=[1, 2, 4, 4]):
        super(PeleeNet, self).__init__()
        # self.cfg = cfg
        self.stage = nn.Sequential()
        self.num_classes = num_classes
        self.num_init_features = num_init_features
        # self.return_features_num_channels = []
        inter_channel = list()
        self.total_filter = list()
        dense_inp = list()

        self.half_growth_rate = int(growthRate / 2)

        self.stages = nn.ModuleList()
        # building stemblock
        self.stages.append(StemBlock(3, num_init_features))

        for i, b_w in enumerate(bottleneck_width):

            inter_channel.append(int(self.half_growth_rate * b_w / 4) * 4)

            if i == 0:
                self.total_filter.append(num_init_features +
                                         growthRate * nDenseBlocks[i])
                dense_inp.append(self.num_init_features)
            else:
                self.total_filter.append(self.total_filter[i - 1] +
                                         growthRate * nDenseBlocks[i])
                dense_inp.append(self.total_filter[i - 1])

            if i == len(nDenseBlocks) - 1:
                with_pooling = False
            else:
                with_pooling = True

            # building middle stageblock
            self.stages.append(
                self._make_dense_transition(dense_inp[i],
                                            self.total_filter[i],
                                            inter_channel[i],
                                            nDenseBlocks[i],
                                            with_pooling=with_pooling))
        # self.return_features_num_channels = [
        #     self.total_filter[0], self.total_filter[1], self.total_filter[3]
        # ]
        # self.return_features_num_channels = self.return_features_num_channels[
        #     self.cfg.MODEL.BACKBONE.OFFSET:]
        self._initialize_weights()

    def _make_dense_transition(self,
                               dense_inp,
                               total_filter,
                               inter_channel,
                               ndenseblocks,
                               with_pooling=True):
        layers = []

        for i in range(ndenseblocks):
            layers.append(
                DenseBlock(dense_inp, inter_channel, self.half_growth_rate))
            dense_inp += self.half_growth_rate * 2

        # Transition Layer without Compression
        layers.append(TransitionBlock(dense_inp, total_filter, with_pooling))

        return nn.Sequential(*layers)

    def forward(self, x):
        stage0_tb = self.stages[0](x)
        stage1_tb = self.stages[1](stage0_tb)
        stage2_tb = self.stages[2](stage1_tb)
        stage3_tb = self.stages[3](stage2_tb)
        stage4_tb = self.stages[4](stage3_tb)
        # out = [stage1_tb, stage2_tb, stage3_tb, stage4_tb]

        return {'res2': stage1_tb, 'res3': stage2_tb, 'res4': stage4_tb}

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')['state_dict']
        model_dict = self.state_dict()
        for i in model_dict:
            print(i)
            if 'head' in i:
                continue
            mi = i.replace('stages.', 'stage.stage_')
            map_i = 'module.' + mi

            if 'fc' in i or 'classifier' in i:
                continue
            if len(self.state_dict()[i].size()) != 0:
                self.state_dict()[i].copy_(param_dict[map_i])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def output_shape(self):
        return {'res2': ShapeSpec(stride=8, channels=128), 
                'res3': ShapeSpec(stride=16, channels=256), 
                # 'res4': ShapeSpec(stride=32, channels=512), 
                'res4': ShapeSpec(stride=32, channels=704)}


if __name__ == '__main__':
    p = PeleeNet(num_classes=1000)
    print(p)
    input = torch.autograd.Variable(torch.ones(1, 3, 224, 224))
    output = p(input)
    for k, v in output.items():
        print(v.shape)
    # p.load_param('pretrain_models/pelee_model_best.pth')
    # print(output.size())
    # torch.save(p.state_dict(), 'pretrain_models/pelee_imagenet.pth')
    # torch.save(p.state_dict(), 'peleenet.pth.tar')