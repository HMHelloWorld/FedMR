#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import datetime
import os

def save_result(data, ylabel, args):
    data = {'base' :data}

    path = './output/{}'.format(args.noniid_case)

    if args.noniid_case != 5:
        file = '{}_{}_{}_{}_{}_lr_{}_{}_frac_{}_{}.txt'.format(args.dataset, args.algorithm, args.model,
                                                                ylabel, args.epochs, args.lr, datetime.datetime.now().strftime(
                "%Y_%m_%d_%H_%M_%S"),args.frac, args.num_users)
    else:
        path += '/{}'.format(args.data_beta)
        file = '{}_{}_{}_{}_{}_lr_{}_{}_frac_{}_{}.txt'.format(args.dataset, args.algorithm,args.model,
                                                                   ylabel, args.epochs, args.lr,
                                                                   datetime.datetime.now().strftime(
                                                                       "%Y_%m_%d_%H_%M_%S"),args.frac, args.num_users)

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path,file), 'a') as f:
        for label in data:
            f.write(label)
            f.write(' ')
            for item in data[label]:
                item1 = str(item)
                f.write(item1)
                f.write(' ')
            f.write('\n')
    print('save finished')
    f.close()

def save_fedmut_result(data, ylabel, args):
    data = {'base' :data}

    path = './output/{}'.format(args.noniid_case)

    if args.noniid_case != 5:
        file = '{}_{}_{}_{}_{}_lr_{}_{}_frac_{}_radius_{}_accrate_{}_bound_{}.txt'.format(args.dataset, args.algorithm, args.model,
                                                                ylabel, args.epochs, args.lr, datetime.datetime.now().strftime(
                "%Y_%m_%d_%H_%M_%S"),args.frac,args.radius,args.mut_acc_rate,args.mut_bound)
    else:
        path += '/{}'.format(args.data_beta)
        file = '{}_{}_{}_{}_{}_lr_{}_{}_frac_{}_radius_{}_accrate_{}_bound_{}.txt'.format(args.dataset, args.algorithm,args.model,
                                                                   ylabel, args.epochs, args.lr,
                                                                   datetime.datetime.now().strftime(
                                                                       "%Y_%m_%d_%H_%M_%S"),args.frac,args.radius,args.mut_acc_rate,args.mut_bound)

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path,file), 'a') as f:
        for label in data:
            f.write(label)
            f.write(' ')
            for item in data[label]:
                item1 = str(item)
                f.write(item1)
                f.write(' ')
            f.write('\n')
    print('save finished')
    f.close()


def save_model(data, ylabel, args):

    path = './output/{}'.format(args.noniid_case)

    if args.noniid_case != 5:
        file = '{}_{}_{}_{}_{}_lr_{}_{}_frac_{}.txt'.format(args.dataset, args.algorithm, args.model,
                                                                ylabel, args.epochs, args.lr, datetime.datetime.now().strftime(
                "%Y_%m_%d_%H_%M_%S"),args.frac)
    else:
        path += '/{}'.format(args.data_beta)
        file = '{}_{}_{}_{}_{}_lr_{}_{}_frac_{}.txt'.format(args.dataset, args.algorithm,args.model,
                                                                   ylabel, args.epochs, args.lr,
                                                                   datetime.datetime.now().strftime(
                                                                       "%Y_%m_%d_%H_%M_%S"),args.frac)

    if not os.path.exists(path):
        os.makedirs(path)

    # with open(os.path.join(path,file), 'a') as f:
    #     for label in data:
    #         f.write(label)
    #         f.write(' ')
    #         for item in data[label]:
    #             item1 = str(item)
    #             f.write(item1)
    #             f.write(' ')
    #         f.write('\n')
    torch.save(data, os.path.join(path,file))
    print('save finished')
    # f.close()
