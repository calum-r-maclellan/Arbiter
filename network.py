

'''
Pytorch implementations of supervised wideresnet and vgg.

@author: calmac
@started: 13/10/21

'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def get_network(network_name, dataset):
    if dataset == 'mnist':
        return network(network_name, num_classes=10, num_channels=1)
    elif dataset == 'svhn':
        return network(network_name, num_classes=10, num_channels=3)
    elif dataset == 'cifar_10':
        return network(network_name, num_classes=10, num_channels=3)
    elif dataset == 'cifar_100':
        return network(network_name, num_classes=100, num_channels=3)
    else:
        raise Exception('dataset is not supported')


def network(network_name, num_classes, num_channels):
    # Wide ResNet
    if network_name == 'wrn':
        return wide_resnet_16_4(num_classes)  
    # VGG
    elif network_name == 'vgg':
        return vgg11_bn(num_classes)
    else:
        raise Exception('network is not supported')


"""
===============================
Wide ResNets 
===============================
Reimplementation of https://github.com/meliketoy/wide-resnet.pytorch
"""

class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()

        self.in_planes = 16
        
        assert (int((depth - 4) % 6) == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        n_stages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, n_stages[0])
        self.layer1 = self._wide_layer(WideBasicBlock, n_stages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasicBlock, n_stages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasicBlock, n_stages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(n_stages[3], momentum=0.9)
        self.fc = nn.Linear(n_stages[3], num_classes)
        
        self.n_features = n_stages[3]

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, hparam=None):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)

        ''' now we have our features, we want to compute their response 
            to the proposed batch size. 
            we do this using a simple matmul() between the batch size and features.
            during training step, hparam remains None. Only during val step do we use it.
        '''
        if hparam is not None:
            assert out.size() == hparam.size(), 'Reparameterised batch size should be same size as features.'
            response = out.data*hparam # need .data here to use features without incurring version RunTimeError.
            out += response

        out = self.fc(out)
        return out


class WideBasicBlock(nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def wide_resnet_16_4(num_classes, dropout=0.0):
    return WideResNet(16, 4, dropout, num_classes)

def wide_resnet_28_10(num_classes, dropout=0.0):
    return WideResNet(28, 10, dropout, num_classes)

def wide_resnet_28_20(num_classes, dropout=0.0):
    return WideResNet(28, 20, dropout, num_classes)



'''
===============================
VGG 
===============================
'''

class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes):
        super(VGG, self).__init__()
        
        self.features = features
        self.n_features = 512
        
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes), 
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x, hparam=None):
        x = self.features(x)
        out = x.view(x.size(0), -1)
        if hparam is not None:
            response = out.data*hparam
            out += response 
        out = self.fc(out)
        return out


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn(num_classes):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes)


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))




''' LeNet: for SVHN.
'''
class LeNet(nn.Module):

    def __init__(self, num_classes, num_channels=1):
        super(LeNet, self).__init__()
        if num_channels == 1:
            self.n_features = 50*4*4
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 50, 5)
            self.fc1 = nn.Linear(4 * 4 * 50, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)
        else:
            self.n_features = 16*5*5
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x, hparam=None):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        if hparam is not None:
            r = out.data*hparam
            out += r 
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

def lenet_5(num_classes, num_channels):
    return LeNet(num_classes, num_channels)

    
