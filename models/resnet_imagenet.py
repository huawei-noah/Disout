#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.




import torch.nn as nn
import math
import sys
sys.path.append("..")

from disout import Disout,LinearScheduler

dploc =  [73, 77, 81, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 173,177, 181, 188, 192, 196, 200, 204, 208, 212]
convloc =[75, 79, 90, 90, 94, 98, 106, 106, 110, 114, 122, 122, 126, 130, 138, 138, 142, 146, 154, 154, 158, 162, 171, 171, 175, 179, 190, 190, 194, 198, 206, 206, 210, 214]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,dist_prob=None,block_size=None,alpha=None,nr_steps=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_disout(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,dist_prob=0.05,block_size=6,alpha=30,nr_steps=5e3):
        super(Bottleneck_disout, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        
        self.disout1=LinearScheduler(Disout(dist_prob=dist_prob,block_size=block_size,alpha=alpha),
                                        start_value=0.,stop_value=dist_prob,nr_steps=nr_steps)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        
        self.disout2=LinearScheduler(Disout(dist_prob=dist_prob,block_size=block_size,alpha=alpha),
                                        start_value=0.,stop_value=dist_prob,nr_steps=nr_steps)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        
        self.disout3=LinearScheduler(Disout(dist_prob=dist_prob,block_size=block_size,alpha=alpha),
                                        start_value=0.,stop_value=dist_prob,nr_steps=nr_steps)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride
        self.disout4=LinearScheduler(Disout(dist_prob=dist_prob,block_size=block_size,alpha=alpha),
                                        start_value=0.,stop_value=dist_prob,nr_steps=nr_steps)

    def forward(self, x):
          
        residual = x

        out = self.conv1(x)       
        out = self.bn1(out)
        out = self.relu(out)
        out=self.disout1(out)
                
        out = self.conv2(out)       
        out = self.bn2(out)
        out = self.relu(out)
        out=self.disout2(out)
                
        out = self.conv3(out)
        out = self.bn3(out)
        out=self.disout3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        residual=self.disout4(residual)
        out += residual
        out = self.relu(out)

        return out


class ResNet_disout(nn.Module):

    def __init__(self, layers, num_classes=1000,dist_prob=0.05,block_size=6,alpha=30,nr_steps=5e3):
        super(ResNet_disout, self).__init__()
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck_disout, 256, layers[2], stride=2,
                                       dist_prob=dist_prob/4,block_size=block_size,alpha=alpha,nr_steps=nr_steps)
        self.layer4 = self._make_layer(Bottleneck_disout, 512, layers[3], stride=2,
                                       dist_prob=dist_prob,block_size=block_size,alpha=alpha,nr_steps=nr_steps)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for name,m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m,nn.BatchNorm2d) and 'bn3'in name:
                m.weight.data.fill_(0)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,dist_prob=0.05,block_size=6,alpha=30,nr_steps=5e3):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            dist_prob=dist_prob,block_size=block_size,alpha=alpha,nr_steps=nr_steps))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                dist_prob=dist_prob,block_size=block_size,alpha=alpha,nr_steps=nr_steps))
        return nn.Sequential(*layers)

    def forward(self, x):
        
        gpu_id = str(x.get_device())
        modulelist=list(self.modules())
        for imodu in range(len(dploc)):
            modulelist[dploc[imodu]].weight_behind[gpu_id]=modulelist[convloc[imodu]].weight.data
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50_disout(dist_prob=0.05,block_size=6,alpha=30,nr_steps=5e3):
    model = ResNet_disout([3, 4, 6, 3],dist_prob=dist_prob,block_size=block_size,alpha=alpha,nr_steps=nr_steps)
    return model



