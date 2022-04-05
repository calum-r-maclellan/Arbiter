'''

hypertrain.py

Main script for training DNNs in parallel with batch size scheduling.

Updates:
- added compatibility for hypergradient descent of the learning rate
- added custom dataloader for retreiving samples from the validation set without needing
to keep redefining all other dataloaders (inefficient).

@latestUpdate: 17/11/21
@dateStarted: 13/10/21
@author: calmac
'''
import time
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from hyper_scheduler import Arbiter
from dataset import get_dataloaders
from network import get_network
from utils import write_dist, write_results, count_correct, convertToBatchSize, convertFromBatchSize 
from sgd_hd import SGDHD 
from adam_hd import AdamHD


parser = argparse.ArgumentParser(description='Parser for running hyperparameter optimisation of the batch size.')
parser.add_argument('--arch', type=str, default='wrn', choices=['lenet', 'vgg', 'wrn'])
parser.add_argument('--dataset', type=str,  default='cifar_10', choices=['svhn','cifar_10','cifar_100'])
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--optim_type', type=str, default='sgd', choices=['sgd','adam'])
parser.add_argument('--lr_scheduler_type', type=str, default='off', choices=['off','step','exp','cosine','hd'])
parser.add_argument('--hd_hyperlr', type=float, default=1e-4, help='learning rate for updating the learning rate in hypergradient descent')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay', type=float, default=0.1)
parser.add_argument('--lr_milestones', type=int, default=[50, 75])
parser.add_argument('--t_0', type=int, default=10)
parser.add_argument('--t_mult', type=int, default=2)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('--alpha_lr', type=float, default=1e-4)
parser.add_argument('--phi_lr', type=float, default=1e-5)
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--n_samples', type=int, default=4,                  help='number of batch size samples we want to pull from phi.')
parser.add_argument('--s_min', type=int, default=8,                      help='smallest possible batch size.')
parser.add_argument('--s_max', type=int, default=512,                    help='maximum possible batch size (s.t. GPU capacity).')
parser.add_argument('--sched_int', type=int, default=1,                  help='scheduling interval (epochs), or how many epochs we allow phi to learn about the dynamics before changing the batch size.')
parser.add_argument('--train_percent', type=float, default=0.8)
parser.add_argument('--save_results', type=bool, default=True)
parser.add_argument('--save_weights', type=bool, default=False)
parser.add_argument('--use_warm', type=bool, default=False)
parser.add_argument('--bs_scheduling', type=bool, default=False)
parser.add_argument('--fixed_batch_sizes', type=int, default=[128, 256, 512], help='assume we start at B=64.')
parser.add_argument('--bs_milestones', type=int, default=[25, 50, 100])
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--model_dir', type=str, default='./checkpoints')
parser.add_argument('--log_root', type=str, default='./logs')

args = parser.parse_args()

def main(args, results_files):
    
    print('Args:', args)
    write_results('{}'.format(args), results_files['sett'])
    
    ''' dataset and dataloaders
    '''
    dataloaders = get_dataloaders(args.dataset, args.batch_size)
    write_results('{}'.format(args.batch_size), results_files['bs'])
    
    ''' INNER SYSTEM:
        DNN with weights w.
    '''
    net = get_network(args.arch, args.dataset).cuda()

    # optimisation:     
    #     - includes possibility of using hypergradient descent to update the learning rate
    #       of the inner system. 
    #     - if no hd wanted, use normal optimiser and lr decay schedule.
    # TODO: get this stuff into a separate function, so we can return the scheduler and 
    #       optimiser without s'eeing all this crap in the main script.
    if args.lr_scheduler_type == 'hd':
        print('\nhd lr decay on ({}).'.format(args.optim_type))
        if args.optim_type == 'adam':
            net_optimiser = AdamHD(net.parameters(), lr=args.lr, weight_decay=args.wd, hypergrad_lr=args.hd_hyperlr)
        else:
            net_optimiser = SGDHD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, hypergrad_lr=args.hd_hyperlr, results_files=results_files)
    else:
        if args.optim_type == 'adam':
          net_optimiser = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
        else:
          net_optimiser = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

        ''' Learning rate scheduler: Choose whether to use hypergradient, or conventional
            methods (e.g. exponential, cosine annealing, warm restarts).
        '''
        if args.lr_scheduler_type == 'exp':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(net_optimiser, gamma=args.lr_decay)
        elif args.lr_scheduler_type == 'step':
            print('\nstep lr decay on.')
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(net_optimiser, milestones=args.milestones, gamma=args.lr_decay)
        elif args.lr_scheduler_type == 'cosine':
            print('\ncosine lr decay on.')
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(net_optimiser, T_0=args.t_0, T_mult=args.t_mult, eta_min=args.lr * 1e-2)
        elif args.lr_scheduler_type == 'off':
            print('\nlr decay off.')
        lr_scheduler = None

    ''' OUTER SYSTEM: 
        batch size scheduler: initialise the model parameters (phi and alpha) 
                              together with their optimiser.
                            : run 'warmup' to bring samples around initial batch size.
    '''
    arbiter = Arbiter(args, n_features=net.n_features, results_files=results_files).cuda()
    arbiter_optimiser = torch.optim.Adam([
                                    {'params':arbiter.parameters(),   'lr':args.phi_lr},
                                    {'params':arbiter.alpha_params(), 'lr':args.alpha_lr},
                                    ], weight_decay=args.wd)
    
    optimisers = {'net':net_optimiser, 'arbiter':arbiter_optimiser}

    ''' misc
    '''
    criterion = nn.CrossEntropyLoss()
    num_train = int(args.train_percent*len(dataloaders['train'].dataset))
    num_val = int(np.round(1-args.train_percent, 1)*len(dataloaders['val'].dataset)) 
    num_test = int(len(dataloaders['test'].dataset))

    ''' warmup: bring samples around current value.
    '''
    if use_warm:
        warmsched_weights = './warmedUp_scheduler_s={}.pth.tar'.format(batch_size)
        if not os.path.exists(warmsched_weights): 
            raise AssertionError('You want to use warmed up but no pretrained weights found.')
        else:
            agent.load_state_dict(torch.load(warmsched_weights))
    else:
        warmup(arbiter, arbiter_optimiser, dataloaders)
        print('done')
    

    best_acc = 0.0
    new_dataloaders = None

    write_results('{}'.format(args.lr), results_files['lrit'])
    write_results('{}'.format(args.lr), results_files['lrep'])

    for epoch in range(args.num_epochs):
        ''' Main loop:
            - now that the samples are lying around where we want them, 
              we can begin moving their distribution in accordance with the val. error
              --> hyperparameter optimisation starts here!
        '''
        if new_dataloaders is not None:
            dataloaders = new_dataloaders
            
        print()
        print('Epoch {} stats:'.format(epoch+1))
        
        ''' 
        ======================
        Training epoch 
        ======================
        '''
        t_start = time.time()
        samples, train_stats  = train(net, arbiter, dataloaders, criterion, optimisers)
        print('epoch took {}s'.format(time.time() - t_start))
        write_results('{}'.format(time.time() - t_start), results_files['times'])

        if args.lr_scheduler_type != 'hd' and lr_scheduler is not None:
          lr_scheduler.step()

        train_acc = 100. * (train_stats['correct'] / num_train)
        print(
              '\tTraining accuracy: {}/{} ({}%)'.format(
              train_stats['correct'], num_train, train_acc
              )
        )
        print('\tTraining loss: {}'.format(train_stats['loss']))
        
        ''' 
        ======================
        Validation epoch 
        ======================
        '''
        val_stats = validation(net, dataloaders['val'], criterion)
        val_acc = 100. * (val_stats['correct'] / num_val)
        print(
              '\tValidation accuracy: {}/{} ({}%)'.format(
              val_stats['correct'], num_val, val_acc
              )
        )
        print('\tValidation loss: {}'.format(val_stats['loss']))
        
        ''' 
        ======================
        Testing epoch 
        ======================
        '''
        test_stats   = test(net, dataloaders['test'], criterion)
        test_acc = 100. * (test_stats['correct'] / num_test)
        print(
              '\tTest accuracy: {}/{} ({}%)'.format(
              test_stats['correct'], num_test, test_acc
              )
        )
        print('\tTest loss: {}\n'.format(test_stats['loss']))

        cur_lr = net_optimiser.param_groups[0]['lr']
        print('end of epoch lr: {}'.format(cur_lr))
        write_results('{}'.format(cur_lr), results_files['lrep'])
        
        ''' select new batch size by asking alpha which sample is most
            likely to be the best. 
            --> check scheduling interval first.
        '''
        if (epoch+1) % args.sched_int == 0:
            alphas = arbiter_optimiser.param_groups[1]['params'][0]
            alphas = F.softmax(alphas, dim=-1)
            alphas = [a.item() for a in alphas]
            samples = [convertToBatchSize(l) for l in samples]
            print('samples: {}'.format(samples))
            print('alphas: {}'.format(alphas))
            sample_best = samples[torch.max(alphas, dim=0)[1]]
            new_batch_size = convertToBatchSize(sample_best)
            print('batch size for the next epoch: {}'.format(new_batch_size))
            new_dataloaders = get_dataloaders(args.dataset, batch_size=new_batch_size)
            assert new_batch_size == new_dataloaders['train'].batch_size, 'Mismatch between batch sizes.'
            write_results('{}'.format(new_batch_size), results_files['bs'])
            write_dist(samples, results_files['epoch-samples'])
            write_dist(alphas, results_files['epoch-alphas'])
            
            ''' reset alphas to setup new search over the next learning period.
            '''
            arbiter_optimiser.param_groups[1]['params'] = arbiter.alpha_params(reset=True)
        

        ''' Fixed scheduling heuristics: 
            check milestones. 
        ''' 
        if args.bs_scheduling:
            if (epoch+1) in args.bs_milestones:
                idx = args.bs_milestones.index(epoch+1)
                old_batch_size = new_dataloaders['train'].batch_size
                new_batch_size = args.fixed_batch_sizes[idx]
                print('batch size changed from {} to {}.'.format(old_batch_size, new_batch_size))
                new_dataloaders = get_dataloaders(dataset, batch_size=new_batch_size)
                write_results('{}'.format(new_batch_size), results_files['bs'])
                
                ''' bring phi samples close to new batch size
                '''
                warmup(arbiter, arbiter_optimiser, new_dataloaders) 
                print('done')

                ''' save all weights for quick restarts
                '''
                # inner system
                torch.save(
                    net.state_dict(), 
                    os.path.join(args.log_root, '{}_{}_valLoss={:4.4f}_valAcc={:4.4f}_epoch={}.pth.tar'.format(args.arch, args.dataset, val_stats['loss'], val_acc, epoch+1))
                  )   
                
                # outer system
                torch.save(
                    arbiter.state_dict(), 
                    os.path.join(log_root, 'scheduler_s={}_epoch={}.pth.tar'.format(new_batch_size, epoch+1))
                    )

        # Write usual performance results to .txt files for analysis 
        if args.save_results:
            write_results('{:.6f}'.format(train_stats['loss']), results_files['trl'])
            write_results('{:.6f}'.format(train_acc), results_files['tra'])
            write_results('{:.6f}'.format(val_stats['loss']), results_files['vl'])
            write_results('{:.6f}'.format(val_acc), results_files['va'])
            write_results('{:.6f}'.format(test_stats['loss']), results_files['tel'])
            write_results('{:.6f}'.format(test_acc), results_files['tea'])
        
        # save weights 
        if val_acc > best_acc:
            best_acc = val_acc
            print('best val acc so far: {}'.format(best_acc))
            write_results('best val acc of {} at epoch  {}'.format(best_acc, epoch+1), results_files['sett'])
            if args.save_weights:
              torch.save(
                net.state_dict(), 
                os.path.join(args.model_dir, '{}_{}_valLoss={:4.4f}_valAcc={:4.4f}_epoch={}.pth.tar'.format(args.arch, args.dataset, val_stats['loss'], val_acc, epoch+1))
              )   
              
       
def train(net, agent, dataloaders, criterion, optimisers):
    
    net.train()   
    arbiter.train()
    net_optimiser = optimisers['net']
    arbiter_optimiser = optimisers['arbiter']
    correct = 0
    loss_list = []
    
    for i, (inputs, targets) in enumerate(dataloaders['train']):

        write_results('{}'.format(dataloaders['train'].batch_size), results_files['bsi'])
        
        net_optimiser.zero_grad()
        arbiter_optimiser.zero_grad()
        custom_dataloader = dataloaders['custom'] # dataloader for requesting samples without needing to reload entire dataset

        ''' Step 1: pass training data through inner model and update params:
            : w --> w*
        '''        
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss_list.append(loss.item())
        correct += count_correct(outputs, targets)
        loss.backward()
        net_optimiser.step() 

        ''' Step 2: sample from batch size scheduler, and request a new sample from dataloader    
        '''
        val_inputs, _ = next(iter(dataloaders['val']))
        samples, s, B, hparam = arbiter(val_inputs.cuda())
        new_data, new_targets = custom_dataloader(sample_size=B)
        new_data, new_targets = new_data.cuda(), new_targets.cuda()

        ''' Step 3: send the new data (of size s_alpha) into w*, together with the reparameterised batch size,
                    compute our features' response, and network output.
        '''
        new_outputs = net(new_data, hparam=hparam)    
        F = criterion(new_outputs, new_targets)
        F.backward()
        arbiter_optimiser.step()
        write_results('{}'.format(F.item()), results_files['F']) 
    
    return samples, {'correct': correct, 'loss': np.mean(loss_list)}


def validation(net, dataloader, criterion):
    net.eval()
    correct = 0
    loss_list = []
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            output = net(data) 
            loss = criterion(output, target)
            loss_list.append(loss.item())
            correct += count_correct(output, target)
    return {'correct': correct, 'loss': np.mean(loss_list)}


def test(net, dataloader, criterion):
    net.eval()
    correct = 0
    loss_list = []
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            output = net(data) 
            loss = criterion(output, target)
            loss_list.append(loss.item())
            correct += count_correct(output, target)
    return {'correct': correct, 'loss': np.mean(loss_list)}


if __name__ == '__main__':
    
    # create files for saving results
    if args.save_weights and not os.path.exists(args.model_dir): os.mkdir(args.model_dir)
    if args.save_results and not os.path.exists(args.log_root): os.mkdir(args.log_root)
    
    ''' TODO come up with a cleaner way to write to files (e.g. csv)
    '''
    settings_txt = open(os.path.join(args.log_root, '_settings.txt'), 'w')
    train_loss_file = open(os.path.join(args.log_root, 'train_loss.txt'), 'w')
    train_acc_file = open(os.path.join(args.log_root, 'train_acc.txt'), 'w')
    val_loss_file = open(os.path.join(args.log_root, 'val_loss.txt'), 'w')
    val_acc_file = open(os.path.join(args.log_root, 'val_acc.txt'), 'w')
    if args.want_to_test:
        test_loss_file = open(os.path.join(args.log_root, 'test_loss.txt'), 'w')
        test_acc_file = open(os.path.join(args.log_root, 'test_acc.txt'), 'w')
    
    logits_file = open(os.path.join(args.log_root, 'logits.txt'), 'w')
    bs_samples = open(os.path.join(args.log_root, 'bs_samples.txt'), 'w')
    l_alpha_file = open(os.path.join(args.log_root, 'l_alpha.txt'), 'w')
    alphas_file = open(os.path.join(args.log_root, 'alphas.txt'),  'w')
    F_file = open(os.path.join(args.log_root, 'F.txt'), 'w')

    s_alpha_file = open(os.path.join(args.log_root, 's_alpha.txt'), 'w')
    batchSizeEpoch_file = open(os.path.join(args.log_root, 'batch_size_epoch.txt'), 'w')
    batchSizeIter_file = open(os.path.join(args.log_root, 'batch_size_iter.txt'), 'w')
    lr_iter_txt    = open(os.path.join(args.log_root, 'lr_iter.txt'),    'w')
    lr_epoch_txt   = open(os.path.join(args.log_root, 'lr_epoch.txt'),   'w')  
    hypergrad_file = open(os.path.join(args.log_root, 'h.txt'), 'w')
    
    epochTime_file = open(os.path.join(args.log_root, 'epoch_times.txt'), 'w')
    end_of_epoch_alphas = open(os.path.join(args.log_root, 'epoch_alphas.txt'), 'w')
    end_of_epoch_samples = open(os.path.join(args.log_root, 'epoch_samples.txt'), 'w')
    
    results_files = {'sett':settings_txt,
                    'trl':train_loss_file, 'tra':train_acc_file,
                     'vl':val_loss_file,'va':val_acc_file,
                     'tel':test_loss_file,'tea':test_acc_file,
                     'l':logits_file, 'samples':bs_samples, 'la':l_alpha_file,
                     'alpha':alphas_file, 'F':F_file, 
                     'sa':s_alpha_file, 'bs':batchSizeEpoch_file, 'bsi':batchSizeIter_file,
                     'times':epochTime_file, 'end-alpha':end_of_epoch_alphas, 'end-samples':end_of_epoch_samples,
                     'lrep':lr_epoch_txt, 'lrit':lr_iter_txt, 'h':hypergrad_file,
                     }
    
    main(args, results_files)


