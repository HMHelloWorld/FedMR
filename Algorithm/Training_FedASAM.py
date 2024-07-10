import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
import random
from models.Fed import Aggregation
from utils.utils import save_result
from models.test import test_img
from models.Update import DatasetSplit
from optimizer.Adabelief import AdaBelief
from utils.sam_minimizers import ASAM


class LocalUpdate_FedASAM(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ensemble_alpha = args.ensemble_alpha
        self.verbose = verbose
        self.mixup=False
        self.mixup_alpha=1.0
        self.rho = args.fedsam_rho
        self.eta = args.fedsam_eta

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
            minimizer = ASAM(optimizer, net, self.rho, self.eta)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                model_output = net(images)
                if self.mixup:
                    model_output, targets_a, targets_b, lam = self.mixup_data(model_output['output'], labels)
                    # predictive_loss = self.mixup_criterion(model_output, targets_a, targets_b, lam)
                    # Ascent Step
                    predictive_loss = self.mixup_criterion(model_output['output'], targets_a, targets_b, lam)
                    predictive_loss.backward()
                    minimizer.ascent_step()

                    # Descent Step
                    loss = self.mixup_criterion(net(images)['output'], targets_a, targets_b, lam)
                    loss.backward()
                    minimizer.descent_step()
                else:
                    # Ascent Step
                    predictive_loss = self.loss_func(model_output['output'], labels)
                    predictive_loss.backward()
                    minimizer.ascent_step()
                    # Descent Step
                    self.loss_func(net(images)['output'], labels).backward()
                    minimizer.descent_step()
                    

                # loss = predictive_loss
                Predict_loss += predictive_loss.item()

                # loss.backward()
                # optimizer.step()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        # net.to('cpu')

        return net.state_dict()
    
    def mixup_data(self, x, y):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.loss_func(pred, y_a) + (1 - lam) * self.loss_func(pred, y_b)


def FedASAM(args, net_glob, dataset_train, dataset_test, dict_users):
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
            local = LocalUpdate_FedASAM(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w = local.train(net=copy.deepcopy(net_glob).to(args.device))

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        if iter % 10 == 9:
            item_acc = test(net_glob, dataset_test, args)
            acc.append(item_acc)

    save_result(acc, 'test_acc', args)



def test(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()
