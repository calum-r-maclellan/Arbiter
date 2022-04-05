
'''
Generator for mapping data to batch size proposals, and reparameterising as 
differentiable logits. 

@latest: 
@startdate: 13/10/21.

@author: calmac
'''


import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os 
from utils import write_results, write_dist, convertToBatchSize

class Arbiter(nn.Module):
   
    def __init__(self, args, n_features, results_files):
        super(Arbiter, self).__init__()
        
        self.args = args
        self.results_files = results_files
        
        if self.args.dataset=='svhn':
            self.n_features = 16*5*5
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, self.args.n_samples)
            self.alphas = Variable(1e-6*torch.randn(self.args.n_samples).cuda(), requires_grad=True)
            self.linear = nn.Linear(1, self.n_features, bias=False)
            if self.args.mapping_grads_off:
                for p in self.linear.parameters():
                    p.requires_grad_(False) 

        elif self.args.dataset=='cifar_10' or self.args.dataset=='cifar_100':
            self.layer1 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    )
            
            self.layer2 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    )
            
            self.layer3 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    )
            
            self.layer4 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    )
            
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, self.args.n_samples)
            self.alphas = Variable(1e-6*torch.randn(self.args.n_samples).cuda(), requires_grad=True)
            self.linear = nn.Linear(1, self.n_features, bias=False)
            if self.args.mapping_grads_off:
                for p in self.linear.parameters():
                    p.requires_grad_(False) 

        self.reset = False
        
    def alpha_params(self, reset=False):
        ''' called by outer_optimiser() for sgd updates.
            : option to reset for initiating a new search.  
        '''
        if reset:
            self.alphas = Variable(1e-6*torch.randn(self.args.n_samples).cuda(), requires_grad=True)
        return [self.alphas]

    def _transform(self, x):
        ''' given logits of shape (B, N), return in shape (N,)
        '''
        x = x.transpose(-1, -2)   # reshape to create samples for each element.
        x = torch.mean(x, dim=1)  # average B samples of each nth element.
        x = torch.sort(x)[0]      # arrange in ascending order. 
        return x


    def forward(self, data, warmup=False):
        ''' pass minibatch of data (current batch size) and generate batch size samples.

            compute two versions of the batch size:
                - s: reparameterised for mapping features to responses (real number).
                - B: expected hyperparameter format (natural number)

            map s to R (f-dim feature space) 
        '''
        if self.args.dataset == 'svhn':
            x = self.conv1(data)
            x = self.conv2(x)
            x = self.pooling(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        elif self.args.dataset=='cifar10' or self.args.dataset=='cifar100':
            x = self.layer1(data)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.pooling(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        
        ''' transform logits to produce samples, and mix to create representative sample (s)
        '''
        samples = self._transform(logits)
        weights = F.softmax(self.alphas, dim=-1)
        s = sum(l*w for l, w in zip(samples, weights))
        B = convertToBatchSize(s)
        s_batch = s.repeat(B, 1)
        s_param = self.linear(s_batch)
        
        if not warmup:
            write_dist([l.item() for l in samples], self.results_files['l'])
            write_dist([convertToBatchSize(l, self.args) for l in samples], self.results_files['samples'])
            write_dist([w.item() for w in weights], self.results_files['alpha'])
            write_results('{}'.format(s.item()), self.results_files['s'])
            write_results('{}'.format(B), self.results_files['B'])
        
        return samples, s, B, s_param



