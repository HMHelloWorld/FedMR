#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import copy

from utils.options import args_parser
from utils.set_seed import set_random_seed
from models.Update import *
from models.Nets import *
from models.Fed import Aggregation
from models.test import test_img
from models.resnetcifar import *
from models import *
from utils.get_dataset import get_dataset
from utils.utils import save_result,save_model
from Algorithm.Training_FedGen import FedGen
from Algorithm.Training_FedMR import FedMR
from Algorithm.Training_FedMR import FedMR_Frozen
from Algorithm.Training_FedMR import FedMR_Partition
from Algorithm.Training_FedIndenp import FedIndep
from Algorithm.Training_FedMut import FedMut
from Algorithm.Training_FedExP import FedExP
from Algorithm.Training_FedASAM import FedASAM

def FedAvg(net_glob, dataset_train, dataset_test, dict_users):

    net_glob.train()

    times = []
    total_time = 0

    # training
    acc = []
    loss = []
    train_loss=[]

    for iter in range(args.epochs):

        print('*'*80)
        print('Round {:3d}'.format(iter))


        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_FedAvg(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        if iter % 10 == 9:
            item_acc,item_loss = test_with_loss(net_glob, dataset_test, args)
            ta,tl = test_with_loss(net_glob, dataset_train, args)
            acc.append(item_acc)
            loss.append(item_loss)
            train_loss.append(tl)

    save_result(acc, 'test_acc', args)
    save_result(loss, 'test_loss', args)
    save_result(train_loss, 'test_train_loss', args)
    save_model(net_glob.state_dict(), 'test_model', args)


def FedProx(net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()

    acc = []

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate_FedProx(args=args, glob_model=net_glob, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        if iter % 10 == 9:
            acc.append(test(net_glob, dataset_test, args))

    save_result(acc, 'test_acc', args)

from utils.clustering import *
from scipy.cluster.hierarchy import linkage
def ClusteredSampling(net_glob, dataset_train, dataset_test, dict_users):

    net_glob.to('cpu')

    n_samples = np.array([len(dict_users[idx]) for idx in dict_users.keys()])
    weights = n_samples / np.sum(n_samples)
    n_sampled = max(int(args.frac * args.num_users), 1)

    gradients = get_gradients('', net_glob, [net_glob] * len(dict_users))

    net_glob.train()

    # training
    acc = []

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        previous_global_model = copy.deepcopy(net_glob)
        clients_models = []
        sampled_clients_for_grad = []

        # GET THE CLIENTS' SIMILARITY MATRIX
        if iter == 0:
            sim_matrix = get_matrix_similarity_from_grads(
                gradients, distance_type=args.sim_type
            )

        # GET THE DENDROGRAM TREE ASSOCIATED
        linkage_matrix = linkage(sim_matrix, "ward")

        distri_clusters = get_clusters_with_alg2(
            linkage_matrix, n_sampled, weights
        )

        w_locals = []
        lens = []
        idxs_users = sample_clients(distri_clusters)
        for idx in idxs_users:
            local = LocalUpdate_ClientSampling(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local_model = local.train(net=copy.deepcopy(net_glob).to(args.device))
            local_model.to('cpu')

            w_locals.append(copy.deepcopy(local_model.state_dict()))
            lens.append(len(dict_users[idx]))

            clients_models.append(copy.deepcopy(local_model))
            sampled_clients_for_grad.append(idx)

            del local_model
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        gradients_i = get_gradients(
            '', previous_global_model, clients_models
        )
        for idx, gradient in zip(sampled_clients_for_grad, gradients_i):
            gradients[idx] = gradient

        sim_matrix = get_matrix_similarity_from_grads_new(
            gradients, distance_type=args.sim_type, idx=idxs_users, metric_matrix=sim_matrix
        )

        net_glob.to(args.device)
        if iter % 10 == 9:
            acc.append(test(net_glob, dataset_test, args))
        net_glob.to('cpu')

        del clients_models

    save_result(acc, 'test_acc', args)




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

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    set_random_seed(args.seed)

    dataset_train, dataset_test, dict_users = get_dataset(args)

    if args.model == 'cnn' and args.dataset == 'femnist':
        net_glob = CNNFashionMnist(args)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args)
    elif args.use_project_head:
        net_glob = ModelFedCon(args.model, args.out_dim, args.num_classes)
    elif 'cifar' in args.dataset and 'cnn' in args.model:
        net_glob = CNNCifar(args)
    elif args.model == 'resnet20' and args.dataset == 'mnist':
        net_glob = ResNet20_mnist(args=args).to(args.device)
    elif args.model == 'resnet20' and (args.dataset == 'fashion-mnist' or args.dataset == 'femnist'):
        net_glob = ResNet20_mnist(args=args).to(args.device)
    elif args.model == 'resnet20' and args.dataset == 'cifar10':
        net_glob = ResNet20_cifar(args=args).to(args.device)
    elif args.model == 'resnet20' and args.dataset == 'cifar100':
        net_glob = ResNet20_cifar(args=args).to(args.device)
    elif 'resnet' in args.model:
        if args.dataset == 'mnist' or args.dataset == 'fashion-mnist' or args.dataset == 'femnist':
            net_glob = ResNet18_MNIST(num_classes = args.num_classes)
        else:
            net_glob = ResNet18_cifar10(num_classes = args.num_classes)
    elif 'cifar' in args.dataset and args.model == 'vgg':
        net_glob = VGG16(args)
    elif 'mnist' in args.dataset and args.model == 'vgg':
        net_glob = VGG16_mnist(args)


    net_glob.to(args.device)
    print(net_glob)

    if args.algorithm == 'FedAvg':
        FedAvg(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedProx':
        FedProx(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'ClusteredSampling':
        ClusteredSampling(net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedGen':
        FedGen(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedMR':
        partition = args.fedmr_partition
        if args.fedmr_frozen_type > 0:
            FedMR_Frozen(args, net_glob, dataset_train, dataset_test, dict_users)
        elif partition == 0:
            FedMR(args, net_glob, dataset_train, dataset_test, dict_users)
        else:
            FedMR_Partition(args, net_glob, dataset_train, dataset_test, dict_users,partition)
    elif args.algorithm == 'FedIndep':
        FedIndep(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedMut':
        FedMut(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedExP':
        FedExP(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'FedASAM':
        FedASAM(args, net_glob, dataset_train, dataset_test, dict_users)



