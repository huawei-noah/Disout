#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.


import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from torch.autograd import Variable
import torch.nn.functional as F
from models.resnet import ResNet_disout

from utils import get_training_dataloader, get_test_dataloader, WarmUpLR


parser = argparse.ArgumentParser()




parser.add_argument('--data_root', type=str,default='/cache/cifar10/')
parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--dist_prob', type=float, default=0.09)
parser.add_argument('--block_size', type=int, default=6)
parser.add_argument('--alpha', type=float, default=5)
args,unparsed = parser.parse_known_args()




def train(epoch):
  
    train_loss = 0.0 
    correct = 0.0
    
    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()
 
        images = Variable(images)
        labels = Variable(labels)
        
        labels = labels.cuda()
        images = images.cuda()
                
        batch_size=images.size(0)
 

    
        optimizer.zero_grad()
        outputs = net(images)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        
        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

    
    print('Train set Epoch: {epoch}  Average_loss: {loss:.4f}, Accuracy: {acc:.4f}'.format(epoch=epoch,
        loss=train_loss / len(training_loader.dataset),
        acc=correct.float() / len(training_loader.dataset)))


def eval(epoch):
    net.eval()

    test_loss = 0.0 
    correct = 0.0

    for (images, labels) in cifar10_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average_loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar10_test_loader.dataset),
        correct.float() / len(cifar10_test_loader.dataset)
    ))

    return correct.float() / len(cifar10_test_loader.dataset)

if __name__ == '__main__':
    

    training_loader = get_training_dataloader(args.data_root,
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    cifar10_test_loader = get_test_dataloader(args.data_root,
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    net = ResNet_disout(depth=56, num_classes=10,dist_prob=args.dist_prob,block_size=args.block_size,alpha=args.alpha,
                        nr_steps=len(training_loader)*args.epochs).cuda()
    
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)


    best_acc = 0.0    
    for epoch in range(1, args.epochs+1):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        train(epoch)
        acc = eval(epoch)                    
        if best_acc < acc:
            best_acc=acc
    print('best_acc:',best_acc)

     

