#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
import random
from models.Fed import Aggregation
from utils.utils import save_result
from utils.utils import save_model
from models.test import test_img
from models.Update import DatasetSplit
from optimizer.Adabelief import AdaBelief


class LocalUpdate_FedMR(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ensemble_alpha = args.ensemble_alpha
        self.verbose = verbose

    def train(self, net):

        net.to(self.args.device)

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                model_output = net(images)
                predictive_loss = self.loss_func(model_output['output'], labels)

                loss = predictive_loss
                Predict_loss += predictive_loss.item()

                loss.backward()
                optimizer.step()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        # net.to('cpu')

        return net.state_dict()

def recombination(w_locals, m):

    w_locals_new = copy.deepcopy(w_locals)

    nr = [i for i in range(m)]

    for k in w_locals[0].keys():
        random.shuffle(nr)
        for i in range(m):
            w_locals_new[i][k] = w_locals[nr[i]][k]

    return w_locals_new

def recombination_partition(w_locals, m, partition):
    is_partition = True

    w_locals_new = copy.deepcopy(w_locals)

    nr = [i for i in range(m)]

    p_idx = 0


    random.shuffle(nr)
    idx = 0.0
    layer_num = len(w_locals[0].keys())
    cnt = 0
    for k in w_locals[0].keys():
        if (partition == 0) or idx >= layer_num * partition*cnt:
            random.shuffle(nr)
            cnt = cnt + 1
        for i in range(m):
            w_locals_new[i][k] = w_locals[nr[i]][k]
        idx = idx + 1.0
    print(idx)
    print(partition)

    return w_locals_new

def FedMR(args, net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    acc = []
    w_locals = []
    sim_arr = []
    loss = []
    train_loss = []

    m = max(int(args.frac * args.num_users), 1)
    for i in range(m):
        w_locals.append(copy.deepcopy(net_glob.state_dict()))

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))


        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for i, idx in enumerate(idxs_users):

            net_glob.load_state_dict(w_locals[i])
            local = LocalUpdate_FedMR(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=net_glob)
            w_locals[i] = copy.deepcopy(w)

        # update global weights
        w_glob = Aggregation(w_locals, None) # Global Model Generation

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        

        if iter % 10 == 9:
            item_acc,item_loss = test_with_loss(net_glob, dataset_test, args)
            tc, tl = test_with_loss(net_glob,dataset_train,args)
            acc.append(item_acc)
            loss.append(item_loss)
            train_loss.append(tl)
            sim_arr.append(sim(args, w_locals))

        if iter >= args.first_stage_bound:
            w_locals = recombination(w_locals, m) # Model Recombination
        else:
            for i in range(len(w_locals)):
                w_locals[i] = copy.deepcopy(w_glob)
        


    save_result(acc, 'test_acc', args)
    save_result(sim_arr, 'sim', args)
    save_result(loss, 'test_loss', args)
    save_result(train_loss, 'test_train_loss', args)
    save_model(net_glob.state_dict(), 'test_model', args)


def FedMR_Partition(args, net_glob, dataset_train, dataset_test, dict_users, partition):
    net_glob.train()

    acc = []
    loss = []
    w_locals = []

    m = max(int(args.frac * args.num_users), 1)
    for i in range(m):
        w_locals.append(copy.deepcopy(net_glob.state_dict()))

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))


        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for i, idx in enumerate(idxs_users):

            net_glob.load_state_dict(w_locals[i])
            local = LocalUpdate_FedMR(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=net_glob)
            w_locals[i] = copy.deepcopy(w)

        # update global weights
        w_glob = Aggregation(w_locals, None) # Global Model Generation

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        item_acc,item_loss = test_with_loss(net_glob, dataset_test, args)

        acc.append(item_acc)
        loss.append(item_loss)
        
        if iter >= args.first_stage_bound:
            w_locals = recombination_partition(w_locals, m, partition) # Model Recombination
        else:
            for i in range(len(w_locals)):
                w_locals[i] = copy.deepcopy(w_glob)


    save_result(acc, 'test_acc', args)
    save_result(loss, 'test_loss', args)


def sim(args,net_glob_arr):
    model_num = int(args.num_users*args.frac)
    sim_tab = [[0 for _ in range(model_num)] for _ in range(model_num)]
    minsum = 10
    subminsum = 10
    sum_sim = 0.0
    for k in range(model_num):
        sim_arr = []
        idx = 0
        # sim_sum = 0.0
        for j in range(k):
            sim = 0.0
            s = 0.0
            dict_a = torch.Tensor(0)
            dict_b = torch.Tensor(0)
            cnt = 0
            for p in net_glob_arr[k].keys():
                a = net_glob_arr[k][p]
                b = net_glob_arr[j][p]
                a = a.view(-1)
                b = b.view(-1)


                if cnt == 0:
                    dict_a = a
                    dict_b = b
                else:
                    dict_a = torch.cat((dict_a, a), dim=0)
                    dict_b = torch.cat((dict_b, b), dim=0)
                
                if cnt % 5 == 0:
                    sub_a = a
                    sub_b = b
                else:
                    sub_a = torch.cat((sub_a, a), dim=0)
                    sub_b = torch.cat((sub_b, b), dim=0)
                    # if not a.equal(b):
                    #     sub_a = torch.cat((sub_a, a), dim=0)
                    #     sub_b = torch.cat((sub_b, b), dim=0)

                if cnt % 5 == 4:
                    s+= F.cosine_similarity(sub_a, sub_b, dim=0)
                cnt += 1
            # print(sim)
            s+= F.cosine_similarity(sub_a, sub_b, dim=0)
            sim = F.cosine_similarity(dict_a, dict_b, dim=0)
            # print (sim)
            sim_arr.append(sim)
            sim_tab[k][j] = sim
            sim_tab[j][k] = sim
            sum_sim += copy.deepcopy(s)
    l = int(len(net_glob_arr[0].keys())/5) + 1.0
    sum_sim /= (45.0*l)
    return sum_sim


def test(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()

def test_with_loss(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item(), loss_test
