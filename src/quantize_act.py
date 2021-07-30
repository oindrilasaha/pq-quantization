# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import time
import math
import argparse
from operator import attrgetter
from bisect import bisect_left

import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn

import models
from data import load_data
from optim import CentroidSGD
from quantization import PQ_act
from utils.training import finetune_centroids, evaluate
from utils.watcher import ActivationWatcher
from utils.dynamic_sampling import dynamic_sampling
from utils.statistics import compute_size
from utils.utils import centroids_from_weights, weight_from_centroids

from models.resnet import *


parser = argparse.ArgumentParser(description='And the bit goes down: Revisiting the quantization of neural networks')

parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet50'],
                    help='Pretrained model to quantize')
parser.add_argument('--block', default='all', type=str,
                    help='Block to quantize (if all, quantizes whole network)')

parser.add_argument('--n-iter', default=100, type=int,
                    help='Number of EM iterations for quantization')
parser.add_argument('--n-activations', default=1, type=int,
                    help='Size of the batch of activations to sample from')

parser.add_argument('--n-centroids-threshold', default=4, type=int,
                    help='Threshold for reducing the number of centroids')
parser.add_argument('--eps', default=1e-8, type=float,
                    help='For empty cluster resolution')

parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                    help='Path to ImageNet dataset')
parser.add_argument('--batch-size', default=1, type=int,
                    help='Batch size for fiuetuning steps')
parser.add_argument('--n-workers', default=20, type=int,
                    help='Number of workers for data loading')

parser.add_argument('--finetune-centroids', default=2500, type=int,
                    help='Number of iterations for layer-wise finetuning of the centroids')
parser.add_argument('--lr-centroids', default=0.005, type=float,
                    help='Learning rate to finetune centroids')
parser.add_argument('--momentum-centroids', default=0.9, type=float,
                    help='Momentum when using SGD')
parser.add_argument('--weight-decay-centroids', default=1e-4, type=float,
                    help='Weight decay')

parser.add_argument('--finetune-whole', default=10000, type=int,
                    help='Number of iterations for global finetuning of the centroids')
parser.add_argument('--lr-whole', default=0.001, type=float,
                    help='Learning rate to finetune classifier')
parser.add_argument('--momentum-whole', default=0.9, type=float,
                    help='Momentum when using SGD')
parser.add_argument('--weight-decay-whole', default=1e-4, type=float,
                    help='Weight decay')
parser.add_argument('--finetune-whole-epochs', default=9, type=int,
                    help='Number of epochs for global finetuning of the centroids')
parser.add_argument('--finetune-whole-step-size', default=3, type=int,
                    help='Learning rate schedule for global finetuning of the centroids')
parser.add_argument('--step3', default=False, type=bool,
                    help='Total finetuning')

parser.add_argument('--restart', default='', type=str,
                    help='Already stored centroids')
parser.add_argument('--save', default='', type=str,
                    help='Path to save the finetuned models')


def main():
    # get arguments
    global args
    args = parser.parse_args()
    args.block = '' if args.block == 'all' else args.block

    # student model to quantize
    student = models.__dict__[args.model](pretrained=True, actquant=True).cuda()
    student.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    # data loading code
    train_loader, test_loader = load_data(data_path=args.data_path, batch_size=args.batch_size, nb_workers=args.n_workers)

    # parameters for the centroids optimizer
    opt_centroids_params_all = []

    # book-keeping for compression statistics (in MB)
    size_uncompressed = compute_size(student)
    size_index = 0
    size_centroids = 0
    size_other = size_uncompressed

    # teacher model
    teacher = models.__dict__[args.model](pretrained=True).cuda()
    teacher.eval()

    # Step 1: iteratively quantize the network layers (quantization + layer-wise centroids distillation)
    print('Step 1: Quantize network')
    t = time.time()
    top_1 = 0

    i=0
    for m,n in student.named_modules():
        if isinstance(n,PQReLU):

            if i<4:
                width=56
                n_centroids = 1024
                block_size = 4
            elif i<8:
                width=28
                n_centroids = 1024
                block_size = 4
            elif i<12:
                width=14
                n_centroids = 512
                block_size = 4
            else:
                width=7
                n_centroids = 512
                block_size = 7

            i+=1

            n_blocks = width*width//block_size

            n_iter_activations = math.ceil(args.n_activations / args.batch_size)
            watcher = ActivationWatcher(student, layer=m)
            in_activations_current = watcher.watch(train_loader, criterion, n_iter_activations)
            in_activations_current = in_activations_current[m]


            # print layer size
            print('Quantizing layer: {}, n_blocks: {}, block size: {}, ' \
                  'centroids: {}'.format(m, n_blocks, block_size, n_centroids))

            # quantizer
            quantizer = PQ_act(in_activations_current, eps=args.eps, n_centroids=n_centroids,
                           n_iter=args.n_iter, n_blocks=n_blocks)
                           # stride=stride, padding=padding, groups=groups)

            print("after PQ")
            # quantize layer
            quantizer.encode()

            n.centroids.data = quantizer.centroids

            top_1 = evaluate(test_loader, student, criterion).item()

            # # book-keeping
            print('Quantizing time: {:.0f}min, Top1 after quantization: {:.2f}\n'.format((time.time() - t) / 60, top_1))
            t = time.time()

            # Step 2: finetune centroids
            print('Finetuning centroids')

            # standard training loop
            lrcen = args.lr_centroids
            if i>3:
                lrcen = args.lr_centroids/2
            if i>7:
                lrcen = args.lr_centroids/4
            if i>11:
                lrcen = args.lr_centroids/8

            centroids_params = {'params': n.centroids}
            opt_centroids_params = [centroids_params]
            opt_centroids_params_all.append(centroids_params)

            optimizer_centroids = torch.optim.SGD(opt_centroids_params, lrcen,
                                momentum=args.momentum_centroids,
                                weight_decay=args.weight_decay_centroids)
            n_iter = args.finetune_centroids
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer_centroids, step_size=1, gamma=0.1)

            for epoch in range(1):
                finetune_centroids(train_loader, student, teacher, criterion, optimizer_centroids, n_iter=n_iter)
                print('done')
                top_1 = evaluate(test_loader, student, criterion)
                scheduler.step()
                print('Epoch: {}, Top1: {:.2f}'.format(epoch, top_1))

            print('After {} iterations with learning rate {}, Top1: {:.2f}'.format(n_iter, lrcen, top_1))

            # book-keeping
            print('Finetuning centroids time: {:.0f}min, Top1 after finetuning centroids: {:.2f}\n'.format((time.time() - t) / 60, top_1))
            
            t = time.time()


    print("End of compression")

    # Step 3: finetune whole network

    if args.step3==True:
        print('Step 3: Finetune whole network')
        t = time.time()

        # custom optimizer
        optimizer_centroids_all = torch.optim.SGD(opt_centroids_params_all, lr=args.lr_whole,
                                          momentum=args.momentum_whole,
                                          weight_decay=args.weight_decay_whole)

        # standard training loop
        n_iter = args.finetune_whole
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_centroids_all, step_size=args.finetune_whole_step_size, gamma=0.1)

        for epoch in range(args.finetune_whole_epochs):
            student.train()
            finetune_centroids(train_loader, student, teacher, criterion, optimizer_centroids_all, n_iter=n_iter)
            top_1 = evaluate(test_loader, student, criterion)
            scheduler.step()
            print('Epoch: {}, Top1: {:.2f}'.format(epoch, top_1))

    print('Finetuning whole network time: {:.0f}min, Top1 after finetuning centroids: {:.2f}\n'.format((time.time() - t) / 60, top_1))


if __name__ == '__main__':
    main()
