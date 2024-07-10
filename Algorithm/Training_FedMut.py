import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
import random
from models.Fed import Aggregation
from utils.utils import save_result, save_fedmut_result, save_model
from models.test import test_img
from models.Update import DatasetSplit
from optimizer.Adabelief import AdaBelief


class LocalUpdate_FedMut(object):
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

def FedMut(args, net_glob, dataset_train, dataset_test, dict_users):
    net_glob.train()
    acc = []
    w_locals = []
    sim_arr = []

    m = max(int(args.frac * args.num_users), 1)
    for i in range(m):
        w_locals.append(copy.deepcopy(net_glob.state_dict()))
    
    delta_list = []
    max_rank = 0
    w_old = copy.deepcopy(net_glob.state_dict())
    w_old_s1 = copy.deepcopy(net_glob.state_dict())

    for iter in range(args.epochs):
        w_old = copy.deepcopy(net_glob.state_dict())
        print('*' * 80)
        print('Round {:3d}'.format(iter))


        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for i, idx in enumerate(idxs_users):

            net_glob.load_state_dict(w_locals[i])
            local = LocalUpdate_FedMut(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=net_glob)
            w_locals[i] = copy.deepcopy(w)

        # update global weights
        w_glob = Aggregation(w_locals, None) # Global Model Generation

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        if iter % 10 == 9:
            acc.append(test(net_glob, dataset_test, args))

        w_delta = FedSub(w_glob, w_old, 1.0)
        rank = delta_rank(args,w_delta)
        if rank > max_rank:
            max_rank = rank
        alpha = args.radius
        w_locals = mutation_spread(args, iter, w_glob, w_old, w_locals, m, w_delta, alpha)
        


    save_fedmut_result(acc, 'test_acc', args)
    # save_model(net_glob.state_dict(), 'test_model', args)
    # save_result(sim_arr, 'sim', args)


def mutation_spread(args, iter, w_glob, w_old, w_locals, m, w_delta, alpha):
    # w_delta = FedSub(w_glob,w_old,(args.radius - args.min_radius) * (1.0 - iter/args.epochs) + args.min_radius)
    # if iter/args.epochs > 0.5:
    #     w_delta = FedSub(w_glob,w_old,(args.radius - args.min_radius) * (1.0 - iter/args.epochs)*2 + args.min_radius)
    # else:
        # w_delta = FedSub(w_glob,w_old,(args.radius - args.min_radius) * (iter/args.epochs)*2 + args.min_radius)
    # w_delta = FedSub(w_glob, w_old, args.radius)


    w_locals_new = []
    ctrl_cmd_list = []
    ctrl_rate = args.mut_acc_rate * (1.0 - min(iter*1.0/args.mut_bound,1.0))
    print (ctrl_rate)

    for k in w_glob.keys():
        ctrl_list = []
        for i in range(0,int(m/2)):
            ctrl = random.random()
            if ctrl > 0.5:
                ctrl_list.append(1.0)
                ctrl_list.append(1.0 * (-1.0 + ctrl_rate))
            else:
                ctrl_list.append(1.0 * (-1.0 + ctrl_rate))
                ctrl_list.append(1.0)
        random.shuffle(ctrl_list)
        ctrl_cmd_list.append(ctrl_list)
    cnt = 0
    for j in range(m):
        w_sub = copy.deepcopy(w_glob)
        if not (cnt == m -1 and m%2 == 1):
            ind = 0
            for k in w_sub.keys():
                w_sub[k] = w_sub[k] + w_delta[k]*ctrl_cmd_list[ind][j]*alpha
                ind += 1
        cnt += 1
        w_locals_new.append(w_sub)


    return w_locals_new


def test(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()


def FedSub(w, w_old, weight):
    w_sub = copy.deepcopy(w)
    for k in w_sub.keys():
        w_sub[k] = (w[k] - w_old[k])*weight

    return w_sub

def delta_rank(args,delta_dict):
    cnt = 0
    dict_a = torch.Tensor(0)
    s = 0
    for p in delta_dict.keys():
        a = delta_dict[p]
        a = a.view(-1)
        if cnt == 0:
            dict_a = a
        else:
            dict_a = torch.cat((dict_a, a), dim=0)
               
        cnt += 1
            # print(sim)
    s = torch.norm(dict_a, dim=0)
    return s