#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch
import time

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate,local_update
from models.Nets import MLP, CNNMnist, CNNCifar, CNNKDD, CNNSWAT, CNNWADI
from models.Fed import FedAvg
from models.test import test_img, eval_test
from data.kdd_data import KDDData
from data.swat_data import SWATData
from data.wadi_data import WADIData


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
    elif args.dataset == 'kdd' or args.dataset == 'swat' or args.dataset == 'wadi':
        print("waiting for loading the data...")
    else:
        exit('Error: unrecognized dataset')
    #img_size = dataset_train[0][0].shape

    # build model
    net_loc =[]
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'kdd':
        net_glob = CNNKDD(args=args).to(args.device)
        for i in range(args.num_users):
            net_loc.append(CNNKDD(args=args).to(args.device))
    elif args.model == 'cnn' and args.dataset == 'swat':
        net_glob = CNNSWAT(args=args).to(args.device)
        for i in range(args.num_users):
            net_loc.append(CNNSWAT(args=args).to(args.device))
    elif args.model == 'cnn' and args.dataset == 'wadi':
        net_glob = CNNWADI(args=args).to(args.device)
        for i in range(args.num_users):
            net_loc.append(CNNWADI(args=args).to(args.device))
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    print("global model initial completed")
    # initial local model, weight and metrics
    w_loc = []
    local_acc = []
    sen_spe = []
    for i in range(args.num_users):
        net_loc[i].train()
        local_acc.append([])
    print("local model initial completed")

    # copy weights
    w_glob = net_glob.state_dict()
    for i in range(args.num_users):
        w_loc.append(net_loc[i].state_dict())

    # training
    loss_train = []
    global_acc_mg = []
    global_acc_ml = []
    for i in range(args.num_users):
        global_acc_ml.append([])
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    sum_time = 0.0
    for iter in range(args.epochs):
        start = time.time()
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        # train local model
        print("=======local clients training their own model========")
        for i in range(args.num_users):
            local_train = local_update(args=args, client_idx=i)
            w, loss = local_train.train(net=copy.deepcopy(net_loc[i]).to(args.device),local=True)
            # update local weights
            w_loc[i] = copy.deepcopy(w)
            # copy weight to net_loc
            net_loc[i].load_state_dict(w_loc[i])
            # test local model
            #ltset = WADIData(fine='WADI', client=i, global_t=False, local_t=True, train=False)
            #ltset = SWATData(fine='SWAT', client=i, global_t=False, local_t=True, train=False)
            ltset = KDDData(fine='KDD', client=i, global_t=False, local_t=True, train=False)
            acc_l, senspe = eval_test(net_loc[i], ltset, args, i)
            #store this round acc
            sen_spe.append(senspe)
            print("round ",iter," local model acc:",acc_l)

        print("=======FL communicaiton========")
        print("start training active user")
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            print("Communication round ",iter," active user is ",idx)
            local = local_update(args=args, client_idx=idx)
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device),local=False)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        print("uploading local model completed.")
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        # test global model
        print("=======Local client evaluate the global model========")
        for i in range(args.num_users):
            #ltset = WADIData(fine='WADI', client=i, global_t=False, local_t=True, train=False)
            #ltset = SWATData(fine='SWAT', client=i, global_t=False, local_t=True, train=False)
            ltset = KDDData(fine='KDD', client=i, global_t=False, local_t=True, train=False)
            acc_g, ss = eval_test(net_glob, ltset, args, i)
            if ss < sen_spe[i]:
                local_acc[i].append(acc_l)
                print("client ",i,"global update dimiss with {:.2f}<{:.2f}".format(ss,sen_spe[i]))
            else:
                local_acc[i].append(acc_g)
                print("client ",i,"global update with {:.2f}>{:.2f}".format(ss, sen_spe[i]))
                w_loc[i] = copy.deepcopy(w_glob)
                net_loc[i].load_state_dict(w_loc[i])
        end = time.time()
        sum_time += end-start
    #final testing
        print("=======Final testing========")
        net_glob.eval()
        sum_acc = 0.0
        '''
        #KDD
        ltset = KDDData(fine='KDD',client=0,global_t=False,local_t=True,train=False)
        gtset = KDDData(fine='KDD',client=0,global_t=True,local_t=False,train=False)
        '''
        '''
        ltset = SWATData(fine='SWAT',client=0,global_t=False,local_t=True,train=False)
        gtset = SWATData(fine='SWAT',client=0,global_t=True,local_t=False,train=False)
        '''
        for i in range(args.num_users):
            net_loc[i].eval()
            #gtset = WADIData(fine='WADI', client=i, global_t=True, local_t=False, train=False)
            #gtset = SWATData(fine='SWAT', client=i, global_t=True, local_t=False, train=False)
            gtset = KDDData(fine='KDD', client=i, global_t=True, local_t=False, train=False)
            localmodel_globaltest_acc, ss = eval_test(net_loc[i], gtset, args, i)
            globalmodel_globaltest_acc, ss = eval_test(net_glob, gtset, args, i)
            #print("Global test accuracy: localmodel 0{:.4f}".format((localmodel_globaltest_acc) * 100))
            global_acc_ml[i].append(localmodel_globaltest_acc * 100)
            sum_acc += globalmodel_globaltest_acc
        #print("Global model globaldata test accuracy:{:.4f}%".format((sum_acc / 5) * 100))
        global_acc_mg.append((sum_acc / args.num_users) * 100)

    print(local_acc)
    print(global_acc_ml)
    global_acc_mg.append(sum_time)
    print(global_acc_mg)
    np.save("results/kdd/6_localtest_ml_acc.npy", local_acc)
    np.save("results/kdd/6_globaltest_ml_acc.npy", global_acc_ml)
    np.save("results/kdd/6_globaltest_mg_acc.npy", global_acc_mg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # final testing
    print("=======Final testing========")
    net_glob.eval()
    sum_acc1 = 0
    sum_acc2 = 0
    sum_acc3 = 0
    '''
    #KDD
    ltset = KDDData(fine='KDD',client=0,global_t=False,local_t=True,train=False)
    gtset = KDDData(fine='KDD',client=0,global_t=True,local_t=False,train=False)
    '''
    '''
    ltset = SWATData(fine='SWAT',client=0,global_t=False,local_t=True,train=False)
    gtset = SWATData(fine='SWAT',client=0,global_t=True,local_t=False,train=False)
    '''
    print("  1.Local Model testing")
    for i in range(args.num_users):
        #ltset = WADIData(fine='WADI',client=i,global_t=False,local_t=True,train=False)
        #gtset = WADIData(fine='WADI',client=i,global_t=True,local_t=False,train=False)
        #ltset = SWATData(fine='SWAT', client=i, global_t=False, local_t=True, train=False)
        #gtset = SWATData(fine='SWAT', client=i, global_t=True, local_t=False, train=False)
        ltset = KDDData(fine='KDD', client=0, global_t=False, local_t=True, train=False)
        gtset = KDDData(fine='KDD', client=0, global_t=True, local_t=False, train=False)
        localmodel_localtest_acc, ss = eval_test(net_loc[i], ltset, args,i)
        localmodel_globaltest_acc, ss = eval_test(net_loc[i], gtset, args,i)
        globalmodel_globaltest_acc, ss = eval_test(net_glob, gtset, args,i)
        print("  Local Model testing")
        print("Local test accuracy: localmodel 0{:.4f}".format((localmodel_localtest_acc)*100))
        print("Global test accuracy: localmodel 0{:.4f}".format((localmodel_globaltest_acc)*100))
        sum_acc1 += localmodel_localtest_acc
        sum_acc2 += localmodel_globaltest_acc
        sum_acc3 += globalmodel_globaltest_acc
    print("  2.Final testing")
    print("Communication completed.")
    print("Local model localdata test average accuracy:{:.4f}%".format((sum_acc1/5)*100))
    print("Local model globaldata test average accuracy:{:.4f}%".format((sum_acc2/5)*100))
    print("Global model globaldata test accuracy:{:.4f}%".format((sum_acc3/5)*100))

