#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from operator import ne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser

from torch.utils.data import DataLoader, Dataset
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from torch import nn, autograd

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print("noniid模式")
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    ###经费
    B = 2000
    
    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    globe_optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    loss_func = nn.CrossEntropyLoss()
    
    ### 初始化每个客户端随机扰动情况
    if args.ppl :
        privacy_preferences = [random.uniform(20,100) for i in range(args.num_users)]
        privacy_preferences.sort(reverse=True)
    
    ### 初始化每个client 数据集以及模型
    client_set_list=[]
    net_local_list = []
    for idx in range(args.num_users):
        client_set=[]
        ldr_train = DataLoader(DatasetSplit(dataset_train, dict_users[idx]), batch_size=args.local_bs, shuffle=True)
        for batch_idx, (images, labels) in enumerate(ldr_train):
            client_set.append((images, labels))
        client_set_list.append(client_set)
        net = copy.deepcopy(net_glob).to(args.device)
        net_local_list.append(net)
        
    print("每个客户端一个epoch的batch",len(client_set_list[0]))
    
    round_count = 0
    for iter in range(args.epochs):
        loss_locals = []
        for round in range(len(client_set_list[0])):
            # 每轮更新模型
            net_glob.zero_grad()
            round_count+=1
            # 模拟只有部分客户端参与
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)


            w_glob = net_glob.state_dict()
            glob_B = B
            ##初始化基础梯度参数 
            grad_dict_update = {}
            for name, parms in net_glob.named_parameters():
                init_tensor = torch.zeros_like(parms)
                grad_dict_update[name]=init_tensor
                
            # 模拟多客户端训练
            for idx in idxs_users:
                net = net_local_list[idx]
                net.load_state_dict(w_glob)
                images, labels = client_set_list[idx][round]
                images, labels = images.to(args.device), labels.to(args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                grad_dict={}
                ## 如何引入了ppl机制
                if args.ppl :
                    local_privacy = privacy_preferences[idx]
                    if glob_B >=local_privacy:
                        local_privacy = 0
                        glob_B -= privacy_preferences[idx]
                    else:
                        local_privacy = local_privacy -glob_B
                        glob_B = 0
                    #print(local_privacy)
                for name, parms in net.named_parameters():
                    grad_dict_update[name]+=parms.grad
                    if args.ppl :
                        if local_privacy!=0:
                            local_std = 1.414*parms.grad/local_privacy
                            #print(local_std)
                            grad_dict_update[name]+= torch.normal(torch.zeros_like(parms),local_std).to(args.device)              
                loss_locals.append(loss)

            for name, parms in net_glob.named_parameters():
                parms.grad = grad_dict_update[name]
            globe_optimizer.step()
            
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            if round_count%10==0:
                print('Round {:3d}, Average loss {:.3f}'.format(round_count, loss_avg))
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

