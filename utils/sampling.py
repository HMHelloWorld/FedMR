#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    return iid(dataset, num_users)


def mnist_noniid(dataset, num_users, case=1):
    num_shards, num_imgs = 100, 600
    return non_iid(dataset, num_users, num_shards, num_imgs, case)


def fashion_mnist_iid(dataset, num_users):
    return iid(dataset, num_users)


def fashion_mnist_noniid(dataset, num_users, case=1):
    num_shards, num_imgs = 100, 600
    return non_iid(dataset, num_users, num_shards, num_imgs, case)


def cifar_iid(dataset, num_users):
    return iid(dataset, num_users)


def cifar_noniid(dataset, num_users, case=1):
    num_shards, num_imgs = 100, 500
    return non_iid(dataset, num_users, num_shards, num_imgs, case)


def cifar100_iid(dataset, num_users):
    return iid(dataset, num_users)


def cifar100_noniid(dataset, num_users, case=1):
    num_shards, num_imgs = 100, 500
    return non_iid(dataset, num_users, num_shards, num_imgs, case)


def svhn_iid(dataset, num_users):
    return iid(dataset, num_users)


def svhn_noniid(dataset, num_users, case=1):
    num_shards, num_imgs = 100, 700
    return non_iid(dataset, num_users, num_shards, num_imgs, case)


def iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    for i in range(num_users):
        dict_users[i] = np.array(list(dict_users[i])).tolist()
    return dict_users


def non_iid(dataset, num_users, num_shards, num_imgs, case=1):
    if case == 1:
        return noniid_ratio_r_label_1(dataset, num_users, num_shards, num_imgs)
    elif case == 2:
        return noniid_label_2(dataset, num_users, int(num_shards * 2), int(num_imgs / 2))
    elif case == 3:
        return noniid_ratio_r_label_1(dataset, num_users, num_shards, num_imgs, ratio=0.8)
    elif case == 4:
        return noniid_ratio_r_label_1(dataset, num_users, num_shards, num_imgs, ratio=0.5)
    else:
        exit('Error: unrecognized noniid case')


def noniid_ratio_r_label_1(dataset, num_users, num_shards, num_imgs, ratio=1):
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:int((rand + ratio) * num_imgs)]),
                                           axis=0)
            random.shuffle(dict_users[i])

    if ratio < 1:
        rest_idxs = np.array([], dtype='int64')
        idx_shard = [i for i in range(num_shards)]
        for i in idx_shard:
            rest_idxs = np.concatenate((rest_idxs, idxs[int((i + ratio) * num_imgs):(i + 1) * num_imgs]), axis=0)
        num_items = int(len(dataset) / num_users * (1 - ratio))
        for i in range(num_users):
            rest_to_add = set(np.random.choice(rest_idxs, num_items, replace=False))
            dict_users[i] = np.concatenate((dict_users[i], list(rest_to_add)), axis=0)
            rest_idxs = list(set(rest_idxs) - rest_to_add)
            random.shuffle(dict_users[i])

    for i in range(num_users):
        dict_users[i] = dict_users[i].tolist()

    return dict_users


def noniid_label_2(dataset, num_users, num_shards, num_imgs):
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for i in range(num_users):
        len_idx_shard = len(idx_shard)
        rand1 = np.random.choice(idx_shard[0:int(len_idx_shard / 2)], 1, replace=False)[0]
        rand2 = np.random.choice(idx_shard[int(len_idx_shard / 2):len_idx_shard], 1, replace=False)[0]
        idx_shard = list(set(idx_shard) - set([rand1, rand2]))
        for rand in [rand1, rand2]:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:int((rand + 1) * num_imgs)]), axis=0)
            random.shuffle(dict_users[i])
    return dict_users


if __name__ == '__main__':

    trans = transforms.Compose([transforms.ToTensor()])
    dataset_train = datasets.SVHN('../data/svhn/', split='train', download=True, transform=trans)
    # trans = transforms.Compose([transforms.ToTensor()])
    # dataset_train = datasets.FashionMNIST('../data/fashion-mnist/', train=True, download=True, transform=trans)
    # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    num = 100
    d = svhn_noniid(dataset_train, num)
    for user_idx in d:
        print(user_idx)
        print([dataset_train[img_idx][1] for img_idx in d[user_idx]])
