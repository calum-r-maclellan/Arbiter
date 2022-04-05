
''' 
Script for running Baydin's learning rate optimisers, sgd_hd.py and adam_hd.py. 

TODO: merge everything into one repo, rather than having everything separate. 

@author: calmac
@date: 16/8/21.

'''

import time
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from network import get_network
from dataset import get_dataloaders
from utils import write_results, count_correct
from sgd_hd import SGDHD
from adam_hd import AdamHD

# TODO: write all these settings to a .txt file.
parser = argparse.ArgumentParser(description='Parser for running hyperparameter optimisation.')
parser.add_argument('--arch', type=str, default='wrn', choices=['vgg', 'wrn'])
parser.add_argument('--dataset', type=str, default='cifar_10', choices=['cifar_10', 'cifar_100', 'svhn', 'xray'])
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--optim_type', type=str, default='sgd', choices=['sgd', 'adam'])
parser.add_argument('--lr_scheduler_type', type=str, default='off', choices=['hd','staircase', 'cyclic', 'off'])
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--step-size', type=int, default=30)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--t_0', type=int, default=10)
parser.add_argument('--t_mult', type=float, default=2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('--hyper-lr', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--model_dir', type=str, default='./checkpoints')
parser.add_argument('--save_results', type=bool, default=True)
parser.add_argument('--save_weights', type=bool, default=False)
parser.add_argument('--log_root', type=str, default='./logs')
parser.add_argument('--tune_beta', type=bool, default=False)
parser.add_argument('--want_to_test', type=bool, default=True, help='evaluate test perf or not.')
parser.add_argument('--train_percent', type=float, default=0.8)


def main(args, results_files):

  torch.manual_seed(args.seed)
  
  device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

  ''' Dataset/Dataloaders '''
  dataloaders = get_dataloaders(args.dataset, args.batch_size, args.train_percent)

  ''' Model '''
  net = get_network(args.arch, args.dataset).cuda()
  n_params = sum(p.numel() for p in net.parameters())
  print('# params: {:4.6f}M'.format(n_params/1e6))
        
  ''' Optimisation '''
  # Loss functions: define for each dataset. 
  num_train = int(args.train_percent*len(dataloaders['train'].dataset))
  num_val = int(np.round(1-args.train_percent, 1)*len(dataloaders['val'].dataset)) 
  num_test = int(len(dataloaders['test'].dataset))
  loss_function = nn.CrossEntropyLoss().cuda()
  
  # Learning rate scheduler: Choose whether to use hypergradient, or conventional
  # methods (e.g. exponential, cosine annealing, warm restarts).
  # TODO: have all this crap in separate file: keep main() clean!
  if args.lr_scheduler_type == 'hd':
      if args.optim_type == 'adam':
          optimizer = AdamHD(net.parameters(), results_files=results_files, lr=args.lr, weight_decay=args.wd, eps=1e-4, hypergrad_lr=args.hyper_lr)
      else:
          optimizer = SGDHD(net.parameters(), results_files=results_files, lr=args.lr, momentum=args.momentum, weight_decay=args.wd, hypergrad_lr=args.hyper_lr)
  else:
      if args.optim_type == 'adam':
          optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd, eps=1e-4)
      else:
          optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

      if args.lr_scheduler_type == 'ed':
          lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
      elif args.lr_scheduler_type == 'staircase':
          lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
      elif args.lr_scheduler_type == 'cyclic':
          lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t_0, T_mult=args.t_mult, eta_min=args.lr * 1e-4)
      elif args.lr_scheduler_type == 'off':
          lr_scheduler = None

  # write initial hyperparameters
  write_results('{:4.6f}'.format(args.lr), results_files['lrit'])
  write_results('{:4.6f}'.format(args.lr), results_files['lrep'])
  
  print('Initial settings:')
  print('\tarchitecture: {}'.format(args.arch))
  print('\tdataset: {}'.format(args.dataset))
  print('\tlearning rate: {}'.format(args.lr))
  print('\tbatch size: {}'.format(args.batch_size))
  print('\thyper-learning rate: {}'.format(args.hyper_lr))

  best_acc = 0.0

  for epoch in range(args.num_epochs):

    print()
    print('===================')
    print('Epoch {} results:\n'.format(epoch+1))

    cur_lr = optimizer.param_groups[0]['lr']
    print('start of epoch learning rate: {:4.6f}'.format(cur_lr))
    print('start of epoch batch size: {}\n'.format(dataloaders['train'].batch_size))
    ''' 
    ======================
    Training epoch 
    ======================
    '''
    t_start = time.time()

    train_stats  = train(net, dataloaders['train'], loss_function, optimizer)
#    if args.lr_scheduler_type != 'hd' or 'hd-mu':
#      lr_scheduler.step()

    print('epoch took {}s'.format(time.time() - t_start))

    train_acc = 100. * (train_stats['correct'] / num_train)
    print(
          '\tTraining accuracy: {}/{} ({}%)'.format(
          train_stats['correct'], num_train, train_acc
          )
    )
    print('\tTraining loss: {}'.format(train_stats['loss']))
    print('\t---')
    
    if args.lr_scheduler_type != 'hd' and lr_scheduler is not None:
        lr_scheduler.step()
        
    
    ''' 
    ======================
    Validation epoch 
    ======================
    '''
    val_stats = validation(net, dataloaders['val'], loss_function)
    val_acc = 100. * (val_stats['correct'] / num_val)
    print(
          '\tValidation accuracy: {}/{} ({}%)'.format(
          val_stats['correct'], num_val, val_acc
          )
    )
    print('\tValidation loss: {}'.format(val_stats['loss']))
    print('\t---')

    ''' 
    ======================
    Testing epoch 
    ======================
    '''
    if args.want_to_test: 
      test_stats   = test(net, dataloaders['test'], loss_function)
      test_acc = 100. * (test_stats['correct'] / num_test)
      print(
            '\tTest accuracy: {}/{} ({}%)'.format(
            test_stats['correct'], num_test, test_acc
            )
      )
      print('\tTest loss: {}\n'.format(test_stats['loss']))
              

    ''' Pull updated learning rate from the optimizer '''
    cur_lr = optimizer.param_groups[0]['lr']
    print('end of epoch learning rate: {:4.6f}'.format(cur_lr))
    print('batch size at next epoch: {}'.format(dataloaders['train'].batch_size))

     # Write results to .txt files for analysis 
    if args.save_results:
        write_results('{:.6f}'.format(train_stats['loss']), results_files['trl'])
        write_results('{:.6f}'.format(val_stats['loss']), results_files['vl'])
        write_results('{:.6f}'.format(train_acc), results_files['tra'])
        write_results('{:.6f}'.format(val_acc), results_files['va'])
        if args.want_to_test:
            write_results('{:.6f}'.format(test_acc), results_files['tea'])
            write_results('{:.6f}'.format(test_stats['loss']), results_files['tel'])
        write_results('{:.6f}'.format(cur_lr), results_files['lrep'])
        

    # save weights 
    if val_acc > best_acc:
        best_acc = val_acc
        print('best val acc so far: {}'.format(best_acc))
        if args.save_weights:
          torch.save(
            net.state_dict(), 
            os.path.join(args.model_dir, 
             'dnn_{}_{}_valLoss={:4.4f}_valAcc={:4.4f}_epoch={}.pth.tar'.format(args.arch, args.dataset, val_stats['loss'], val_acc, epoch+1))
          )   
 

def train(net, dataloader, loss_function, optimizer, use_val_grads=False):
    '''
        option to use training gradients (Baydin) or validation gradients
        to compute hypergradient.
        TODO: which is more effective?
    '''
    net.train()
    correct = 0
    loss_list = []
    for i, (data, target) in enumerate(dataloader):
        # forward pass and compute loss
        data, target = data.cuda(), target.cuda()
        output = net(data)
        loss = loss_function(output, target)
        # compute gradients and backprop
        loss.backward(create_graph=True)
        # update parameters and zero grads for next iter
        optimizer.step()
        optimizer.zero_grad()
        # record loss and compute performance
        loss_list.append(loss.item())
        correct += count_correct(output, target)
    return {'correct': correct, 'loss': np.mean(loss_list)}

def validation(net, dataloader, loss_function):
    net.eval()
    correct = 0
    loss_list = []
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            output = net(data) 
            loss = loss_function(output, target)
            loss_list.append(loss.item())
            correct += count_correct(output, target)
    return {'correct': correct, 'loss': np.mean(loss_list)}

def test(net, dataloader, loss_function):
    net.eval()
    correct = 0
    loss_list = []
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            output = net(data) 
            loss = loss_function(output, target)
            loss_list.append(loss.item())
            correct += count_correct(output, target)
    return {'correct': correct, 'loss': np.mean(loss_list)}


if __name__ == '__main__':
    
    args = parser.parse_args()
    
    # create files for saving results
    if args.save_weights and not os.path.exists(args.model_dir): os.mkdir(args.model_dir)
    if args.save_results and not os.path.exists(args.log_root): os.mkdir(args.log_root)
    
    train_loss_file = open(os.path.join(args.log_root, 'train_loss.txt'), 'w')
    train_acc_file = open(os.path.join(args.log_root, 'train_acc.txt'), 'w')
    val_loss_file = open(os.path.join(args.log_root, 'val_loss.txt'), 'w')
    val_acc_file = open(os.path.join(args.log_root, 'val_acc.txt'), 'w')
    test_loss_file = open(os.path.join(args.log_root, 'test_loss.txt'), 'w')
    test_acc_file = open(os.path.join(args.log_root, 'test_acc.txt'), 'w')
    hypergrad_file = open(os.path.join(args.log_root, 'hypergrad.txt'), 'w')
    lr_epoch_file = open(os.path.join(args.log_root, 'lr_epoch.txt'), 'w')
    lr_iter_file = open(os.path.join(args.log_root, 'lr_iter.txt'), 'w')

    results_files = {'trl':train_loss_file, 'tra':train_acc_file,
                     'vl':val_loss_file,'va':val_acc_file,
                     'tel':test_loss_file,'tea':test_acc_file,
                     'h':hypergrad_file, 
                     'lrep':lr_epoch_file, 'lrit':lr_iter_file
                     }
    
    main(args, results_files)

