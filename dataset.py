
'''

dataset.py

Script for loading benchmark datasets, including:
  - MNIST, SVHN, CIFAR-10, CIFAR-100.

@author: calmac

'''


import os 
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from custom_dataloader import CustomDataLoader_CIFAR, CustomDataLoader_SVHN


data_loc = '/home/hsijcr/calummac/cifar-data'

def get_dataloaders(dataset, batch_size, train_percent=0.8):
    if dataset == 'mnist':
        return mnist_data_loader(batch_size, train_percent)
    elif dataset == 'cifar_10':
        return cifar_10_data_loader(batch_size, train_percent)
    elif dataset == 'cifar_100':
        return cifar_100_data_loader(batch_size, train_percent)
    elif dataset == 'svhn':
        return svhn_data_loader(batch_size, train_percent)
    else:
        raise Exception('dataset is not supported')


def mnist_data_loader(batch_size, train_percent):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    ''' datasets
    '''
    train_dataset = datasets.MNIST(root=data_loc, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_loc, train=False, transform=transform, )
    
    ''' dataloaders 
    '''
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(train_percent * num_train))
    # training
    train_loader = DataLoader(
      train_dataset, batch_size=batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=False, num_workers=1) 
    # validation data
    val_loader = DataLoader(
      train_dataset, batch_size=batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=False, num_workers=1)
    # test data 
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=1)

    custom_loader = CustomDataLoader_MNIST(train_dataset, transform, train_percent)

    return {'train':train_loader, 'val':val_loader, 'test':test_loader, 'custom':custom_loader}
  
def svhn_data_loader(batch_size, train_percent):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    ''' datasets
    '''
    train_dataset = datasets.SVHN(root=data_loc, split='train', download=True, transform=transform)
    test_dataset = datasets.SVHN(root=data_loc, split='test', transform=transform)

    ''' dataloaders 
    '''
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(train_percent * num_train))
    # training
    train_loader = DataLoader(
      train_dataset, batch_size=batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=False, num_workers=1) 
    # validation data
    val_loader = DataLoader(
      train_dataset, batch_size=batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=False, num_workers=1)
    # test data 
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=1)

    custom_loader = CustomDataLoader_SVHN(train_dataset, transform, train_percent)

    return {'train':train_loader, 'val':val_loader, 'test':test_loader, 'custom':custom_loader}
    


def cifar_10_data_loader(batch_size, train_percent):
    # transforms
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    
    # datasets
    train_dataset = datasets.CIFAR10(root=data_loc, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_loc, train=False, transform=transform_test)
    
    # dataloaders with validation/hessian subsets of the training dataset
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(train_percent * num_train))

    train_loader = DataLoader(
      train_dataset, batch_size=batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=False, num_workers=1) 

    # validation data
    val_loader = DataLoader(
      train_dataset, batch_size=batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=False, num_workers=1)

    # test data 
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=1)

    ''' custom dataloader for the scheduler: 
          enables us to query N samples from the validation dataset without
          having to reload full dataset and create fresh new dataloaders every
          iteration.
    '''
    custom_loader = CustomDataLoader_CIFAR(train_dataset, transform_train, train_percent)

    return {'train':train_loader, 'val':val_loader, 'test':test_loader, 'custom':custom_loader}
  

def cifar_100_data_loader(batch_size, train_percent):
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
          transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), ])
    
    '''datasets '''
    train_dataset = datasets.CIFAR100(root=data_loc, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=data_loc, train=False, transform=transform_test)

    ''' dataloaders '''
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(train_percent * num_train))

    train_loader = DataLoader(
      train_dataset, batch_size=batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=False, num_workers=1) 

    # validation data
    val_loader = DataLoader(
      train_dataset, batch_size=batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=False, num_workers=1)

    # test data 
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=1)
    
    # custom datalaoder: for querying N samples from the validation set. 
    custom_loader = CustomDataLoader_CIFAR(train_dataset, transform_train, train_percent)

    return {'train':train_loader, 'val':val_loader, 'test':test_loader, 'custom':custom_loader}
  