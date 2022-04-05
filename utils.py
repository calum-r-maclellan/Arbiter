

'''
utils.py

Useful functions to supplement the main codes.

@author: calmac
'''

import math
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as ag
import torch.nn.functional as F
import matplotlib.pyplot as plt

def warmup(arbiter, optimiser, dataloaders, save=False):
    print('\nwarming up...')
    B_target = dataloaders['train'].batch_size
    huber = nn.SmoothL1Loss().cuda()
    temp_hyperlr = 1e-4
    optimiser.param_groups[0]['lr'] = optimiser.param_groups[1]['lr'] = temp_hyperlr
    s_target = convertFromBatchSize(B_target).cuda()  
    arbiter.train()
    for j in range(10000): # arbitrarily high iteration length: usually only takes around 1 to 2 epochs (2000 iters at most)
        inputs, _ = next(iter(dataloaders['val']))
        optimiser.zero_grad()
        _, s_pred, B_pred, _ = arbiter(inputs.cuda(), warmup=True)
        distance = huber(s_pred, s_target)
        distance.backward()
        optimiser.step()
        print(B_pred)
        if np.abs(B_pred - B_target) < 2:
            if save:
                torch.save(arbiter.state_dict(), 
                  os.path.join(args.root_path, './warmedUp_scheduler_s={}.pth.tar'.format(B_target))
                )
            optimiser.param_groups[0]['lr'] = phi_lr
            optimiser.param_groups[1]['lr'] = alpha_lr
            optimiser.param_groups[1]['params'] = arbiter.alpha_params(reset=True)
            break
    assert optimiser.param_groups[0]['lr'] == phi_lr, 'Phi has the wrong hyper lr.'
    assert optimiser.param_groups[1]['lr'] == alpha_lr, 'Alpha has the wrong hyper lr.'
    if any(abs(optimiser.param_groups[1]['params'][0]) > 1e-4) == True:
        raise AssertionError ('alphas havent been reset. Make sure they are before starting training.')



def logit(x):
    ''' note: torch.log() is the NATURAL log (not base 2, 10, etc...). 
    '''
    return torch.log(x) - torch.log(1-x)

def convertFromBatchSize(batch_size, args, eps=1e-3):
    """Stretched logit function: 
            Maps batch size lying in N (min, max) to R,
            where (min, max) constrain the possible values the batch size can take.
            (subject to user's GPU capacity)
            -> eps: prevents normalised batch size from equalling 1 (numerical error)
    """
    bs = torch.tensor(np.float(batch_size))
    
    ''' failsafes below for preventing numerical errors
    '''
    if batch_size == args.s_min:
        bs += eps
    elif batch_size == args.s_max:
        bs -= eps
        
#    s_min = torch.tensor(np.float(args.s_min))
#    s_max = torch.tensor(np.float(args.s_max))
    
    return logit((bs - args.s_min) / (args.s_max - args.s_min))


def convertToBatchSize(l, args):
    ''' given logit lying in R, convert to batch size to lie in N. 
        default min and max, but could be changed s.t. GPU capacity.
    '''
    x = (args.s_max - args.s_min)*torch.sigmoid(l) + args.s_min
    return int(np.floor(x.item()))

def init_weights(m):
    if type(m) == nn.Conv2d:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
    elif type(m) == nn.BatchNorm2d:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def count_correct(output, target):
    probs = F.softmax(output,dim=1)
    _, pred = torch.max(probs.data, 1)
    correct = (pred == target).sum().item()
    return correct

def read_results(file):
    with open(file) as f:
        array=[]
        for line in f.readlines():
            array.append(np.array(line).astype(np.float))
        f.close()
    return array
   
  
def write_dist(stuff, file):
    if len(stuff)==4:
        info = str(stuff[0])+' '+str(stuff[1])+' '+str(stuff[2])+' '+str(stuff[3])
    elif len(stuff)==10:
        info = str(stuff[0])+' '+str(stuff[1])+' '+str(stuff[2])+' '+str(stuff[3])+' '+str(stuff[4])+' '+str(stuff[5])+' '+str(stuff[6])+' '+str(stuff[7])+' '+str(stuff[8])+' '+str(stuff[9])
    file.write(info+'\n')
    file.flush()
    
    
def write_results(stuff, file):
    file.write(stuff+'\n')
    file.flush()

def plot_inset(epoch_results, iter_results):
    x = epoch_results
    xi = iter_results
    fig, axes = plt.subplots(1, 3)
    for ax in axes.flat:
        ax.plot(x)                # plot the epoch data
        ax.inset_axes([xi])  # plot the iteration data
    plt.show()
        
        
def plot_results(file, result_name, n_iters, save_fig=False, iter_result=False):
    results = read_results(file)
    fig = plt.figure()
    if iter_result:
        plt.plot(results[:n_iters], 'b')
        plt.xlabel('Iteration')
    else: 
        plt.plot(results, 'b')
        plt.xlabel('Epoch')
    plt.ylabel('Batch size')
    yint = range(min(results), math.ceil(max(results))+1)
    plt.yticks(yint)
#    plt.show()
    if save_fig:
        plt.savefig(os.path.join('./{}.png'.format(result_name)))
        


def clear_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()


def count_params(net):
    ''' counts the number of parameters, then returns as a factor of 1M '''
    return sum(p.numel() for p in net.parameters())/1e6



