#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.




import argparse
import os
import shutil
import time
import math
import warnings
import models

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from disout import Disout,LinearScheduler
from models.resnet_imagenet import resnet50_disout


parser = argparse.ArgumentParser(description='PyTorch Resnet')
parser.add_argument('--data', metavar='DIR',default='/cache/tyh/imagenet/',help='path to dataset')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=540, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--batch-size', default=1024, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.4, type=float,metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1000, type=int,metavar='N', help='print frequency (default: 10)')
parser.add_argument('--manual-seed', default=0, type=int, metavar='N',help='manual seed (default: 0)')


parser.add_argument('--dist_prob', default=0.07, type=float)
parser.add_argument('--block_size', default=7, type=int)
parser.add_argument('--alpha', default=1, type=float)



args,unparsed = parser.parse_known_args()




warnings.filterwarnings("ignore")



torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)





best_prec1 = 0


def main():
    global args, best_prec1

    
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    train_set = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    val_set = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Scale(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    

    
    model=resnet50_disout(dist_prob=args.dist_prob,block_size=args.block_size,alpha=args.alpha,nr_steps=len(train_loader)*args.epochs)
    model = torch.nn.DataParallel(model).cuda()
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay,nesterov=True)




    cudnn.benchmark = True


    namlist=[]
    modulelist=[]

    for nam,module in model.named_modules():
        namlist.append(nam)
        modulelist.append(module)
            

    num_module=len(modulelist)
    dploc=[]
    convloc=[]
    for idb in range(num_module):
        if isinstance(modulelist[idb],Disout):
            dploc.append(idb-1)
            for iconv in range(idb,num_module):
                if isinstance (modulelist[iconv],nn.Conv2d) and not('downsample' in namlist[iconv]):
                    convloc.append(iconv-1)
                    break
    print(len(dploc),len(convloc))
    print('dploc:',dploc)
    print('convloc:',convloc)

    for epoch in range(0, args.epochs):

        tr_prec1, tr_prec5, loss, lr = train(train_loader, model, criterion, optimizer, epoch)

        val_prec1, val_prec5 = validate(val_loader, model, criterion,epoch)

        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)

    print("best_prec1:",best_prec1)

    
    return


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    learned_module_list = []


    model.train()
    running_lr = None
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        
        for module in model.modules():
            if isinstance(module,Disout):
                module.weight_behind={}
  
        for module in model.modules():
            if isinstance(module,LinearScheduler):
                module.step()
        
        progress = float(epoch * len(train_loader) + i) / (args.epochs * len(train_loader))
        args.progress = progress
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,nBatch=len(train_loader))
        if running_lr is None:
            running_lr = lr

 
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)


        output = model(input_var)

        

        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()


    print('Epoch: [{epoch}] lr: {lr: .6f}  * Prec@1_train {top1.avg:.3f} Prec@5_train {top5.avg:.3f} Loss_train {losses.avg:.4f}'.format(epoch=epoch,lr=lr,top1=top1, top5=top5,losses=losses))

    return 100. - top1.avg, 100. - top5.avg, losses.avg, running_lr


def validate(val_loader, model, criterion,epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            output = model(input_var)
            loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))


        batch_time.update(time.time() - end)
        end = time.time()

#         if i % args.print_freq == 0:
#             print('Test: [{0}/{1}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                       i, len(val_loader), batch_time=batch_time, loss=losses,
#                       top1=top1, top5=top5))
    print('Test Epoch: [{epoch}] * Prec@1_test {top1.avg:.3f} Prec@5_test {top5.avg:.3f} Loss_test {losses.avg:.4f}'.format(epoch=epoch,top1=top1, top5=top5,losses=losses))

    return top1.avg, top5.avg





class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args, batch=None,nBatch=None):
  
    lr, decay_rate = args.lr, 0.1
    if epoch >= args.epochs * 0.93:
        lr *= decay_rate**3
    elif epoch >= args.epochs * 0.74:
        lr *= decay_rate**2
    elif epoch >= args.epochs * 0.45:
        lr *= decay_rate
    else: 
        lr=lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
