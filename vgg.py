import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F


def vgg(cfg, input_channels, batch_norm=False):
    layers = []
    in_channels = input_channels
    stage = 1
    for v in cfg:
        if v == 'M':
            stage += 1
            if stage == 6:
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            else:
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        else:
            if stage == 6:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return layers


class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.cfg = {'tun': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                    'tun_ex': [512, 512, 512]}  # outputchannels M-->Merge
        self.extract = [3, 8, 15, 22, 29]  # without batchNorm here [3, 8, 15, 22, 29]
        self.extract_ex = [5]
        self.base = nn.ModuleList(vgg(self.cfg['tun'], 3))
        self.base_ex = vgg_ex(self.cfg['tun_ex'], 512)

        # for k in range(len(self.base)):
        #   print(self.base[k], k)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_pretrained_model(self, model):
        self.base.load_state_dict(model)

    def forward(self, x, multi=0):
        tmp_x = []
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.extract:
                tmp_x.append(x)
        x = self.base_ex(x)
        tmp_x.append(x)
        if multi == 1:
            tmp_y = []
            tmp_y.append(tmp_x[0])
            return tmp_y
        else:
            return tmp_x


class vgg_ex(nn.Module):
    def __init__(self, cfg, incs=512, padding=1, dilation=1):
        super(vgg_ex, self).__init__()
        self.cfg = cfg
        layers = []
        for v in self.cfg:
            conv2d = nn.Conv2d(incs, v, kernel_size=3, padding=padding, dilation=dilation, bias=False)
            layers += [conv2d, nn.ReLU(inplace=True)]
            incs = v
        self.ex = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.ex(x)
        return x

if __name__== '__main__':
    im=torch.randn(1,3,256,256)
    net= vgg16()
    output= net(im)
    for i in output:
        print(i.size())