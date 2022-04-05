

'''
custom_dataloader.py

Designed to extract N samples (specified by batch size scheduler) from a pre-defined
dataset. Idea is to prevent the need to keep redefining dataloaders for new batch sizes 
in the conventional way, cutting down on time and effort.

@started: 17/11/21
@author: calmac
'''

import os 
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils import data 

class CustomDataLoader_CIFAR(object):
    ''' given a dataset and its transform, create a dataloader that can pull N samples from 
        this dataset, where N = S_alpha the batch size predicted by our generator.
        This lets us simply sample from the same dataset (i.e. validation)
        instead of reloading fresh dataloaders each time...(expensive!!)

        returns:
          : minibatch of validation data ("S_alpha" number of samples)
    '''
    def __init__(self, dataset, transform, split):
        # super(CustomDataLoader, self).__init__()

        self.num_train = len(dataset)
        self.val_start = int(np.floor(split * self.num_train))
        self.data = dataset.data
        self.targets = dataset.targets
        self.transform = transform

    def __call__(self, sample_size):

        # choose N==S_alpha indices at random from a list of indices spanning the 
        # validation dataset (i.e. 40000 -> 50000).
        idx = list(np.random.randint(low=self.val_start, high=self.num_train, size=sample_size))
        
        # pull out the data at those indices at store in ndarray
        X = np.array([self.data[i] for i in idx])
        labels = torch.from_numpy(np.array([self.targets[i] for i in idx]))

        # transform each data instance for using in DNN
        for i in range(sample_size):
          xi = Image.fromarray(X[i,:])
          xi = self.transform(xi).unsqueeze(0)
          if i==0:
            inputs = xi
          else:
            inputs = torch.cat((inputs,xi), dim=0)

        return inputs, labels

class CustomDataLoader_MNIST(object):
  
    def __init__(self, dataset, transform, split):

        self.num_train = len(dataset)
        self.val_start = int(np.floor(split * self.num_train))
        self.data = dataset.data
        self.targets = dataset.targets
        self.transform = transform

    def __call__(self, sample_size):

        idx = list(np.random.randint(low=self.val_start, high=self.num_train, size=sample_size))
        # X = np.array([self.data[i] for i in idx]) # works with cifar10/100, not with mnist
        X = [np.array(self.data[i]) for i in idx] # works with mnist, does it with cifar10?
        labels = torch.from_numpy(np.array([self.targets[i] for i in idx]))

        for i in range(sample_size):
          # xi = Image.fromarray(X[i,:]) # worked with cifar10/100, not with mnist 
          xi = Image.fromarray(X[i])   # works with mnist: does it with cifar10?
          xi = self.transform(xi).unsqueeze(0)
          if i==0:
            inputs = xi
          else:
             inputs = torch.cat((inputs,xi), dim=0)

        return inputs, labels


class CustomDataLoader_SVHN(object):
  
    def __init__(self, dataset, transform, split):

        self.num_train = len(dataset)
        self.val_start = int(np.floor(split * self.num_train))
        imgs=[]
        lbls=[]
        for i in range(self.num_train):
          x, y = dataset[i][0], dataset[i][1]
          imgs.append(x)
          lbls.append(y)
        self.data = imgs
        self.targets = lbls
        self.transform = transform

    def __call__(self, sample_size):

        idx = list(np.random.randint(low=self.val_start, high=self.num_train, size=sample_size))
        # X = np.array([self.data[i] for i in idx]) # works with cifar10/100, not with mnist
        # X = [np.array(self.data[i]) for i in idx] # works with mnist, does it with cifar10?
        X = [self.data[i] for i in idx] # works for SVHN
        labels = torch.from_numpy(np.array([self.targets[i] for i in idx]))

        for i in range(sample_size):
          # xi = Image.fromarray(X[i,:]) # worked with cifar10/100, not with mnist 
          # xi = Image.fromarray(X[i])   # works with mnist: does it with cifar10?
          # xi = self.transform(X[i]).unsqueeze(0) # already a PIL image, so just apply transform and crack on
          xi = X[i].unsqueeze(0)
          if i==0:
            inputs = xi
          else:
            inputs = torch.cat((inputs,xi), dim=0)

        return inputs, labels


