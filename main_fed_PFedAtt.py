#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import torchvision
from torchvision import datasets, transforms, models
import torch
from torch import nn
from sklearn.metrics.pairwise import pairwise_distances

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar100_iid, cifar100_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FedAtt
from models.test import test_img

def similar(w_i,w_j):
    a = copy.deepcopy(w_i)
    b = copy.deepcopy(w_j)
    sum_e = 0
    for k in a.keys():
        e = (a[k] - b[k]) * (a[k] - b[k])
        e = e.numpy().sum()
        sum_e = sum_e + e
    c = 1 - np.exp(-sum_e)
    return c

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'cifar100':
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        transform_train = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        dataset_train = torchvision.datasets.CIFAR100(root='../data/cifar100', train=True, download=True,transform=transform_train)
        dataset_test = torchvision.datasets.CIFAR100(root='../data/cifar100', train=False, download=True,transform=transform_test)
        if args.iid:
            dict_users = cifar100_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar100_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and (args.dataset == 'cifar' or args.dataset == 'cifar100'):
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar100':
        net = models.resnet34(pretrained=False)  # pretrained=False or True 不重要
        fc_inputs = net.fc.in_features  # 保持与前面第一步中的代码一致
        net.fc = nn.Sequential(  #
            nn.Linear(fc_inputs, 100),  #
            nn.LogSoftmax(dim=1)
        )
        net_loc = []
        net_glob = net.to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    loc_sim = []
    top_k = 50

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        loc_sim = [0.0 for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        #cal similar
        for i in range(args.num_users):
            sum_sim = 0
            for idx in idxs_users:
                if i != idx:
                    s = similar(w_locals[i],w_locals[idx])
                    sum_sim += s
            loc_sim[i] = sum_sim

        # top_k
        idxs_simsort = np.vstack((range(args.num_users), loc_sim))
        idxs_simsort = idxs_simsort[:, idxs_simsort[1, :].argsort()]
        topk_users = idxs_simsort[0, (args.num_users-top_k):args.num_users + 1]
        attn_users = idxs_simsort[1, (args.num_users - top_k):args.num_users + 1]

        topk_users = list(map(int,topk_users))
        # update global weights
        up_w = []
        for i in topk_users:
            up_w.append(copy.deepcopy(w_locals[i]))
        w_glob = FedAvg(up_w)
        #w_glob = FedAtt(up_w,attn_users)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

