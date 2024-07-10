import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
import random
from models.Fed import Aggregation,Sub,Mul,Div,Add
from utils.utils import save_result
from models.test import test_img,branchy_test_img
from models.Update import DatasetSplit
from optimizer.Adabelief import AdaBelief
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def FedExP(args, net_glob, dataset_train, dataset_test, dict_users):

    net_glob.train()

    times = []
    total_time = 0

    # training
    acc = []
    loss = []
    train_loss=[]

    grad_norm_avg_running = 0

    w_old = copy.deepcopy(net_glob.state_dict())
    p = np.zeros((args.num_users))

    for i in range(args.num_users):
        p[i] = len(dict_users[i])
    
    p = p/np.sum(p)
    d = parameters_to_vector(net_glob.parameters()).numel()

    w_vec_estimate = parameters_to_vector(net_glob.parameters())
    

    for iter in range(args.epochs):

        print('*'*80)
        print('Round {:3d}'.format(iter))


        w_locals = []
        lens = []
        grad_norm_sum = 0
        p_sum =0
        grad_avg = copy.deepcopy(net_glob.state_dict())
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        tag = 0
        # pre_grad = parameters_to_vector(net_glob.parameters())
        for idx in idxs_users:
            local = LocalUpdate_FedExP(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, grad_local = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            w_grad = Sub(w, net_glob.state_dict())
            grad = parameters_to_vector(grad_local) - parameters_to_vector(net_glob.parameters())
            grad_norm_sum += p[idx]*torch.linalg.norm(grad)**2
            w_grad = Mul(w_grad,p[idx])
            if (tag == 0):
                grad_avg = w_grad
            else:
                grad_avg = Add(grad_avg,w_grad)
            p_sum += p[idx]
            lens.append(len(dict_users[idx]))
            tag += 1
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        with torch.no_grad():
            grad_avg = Div(grad_avg,p_sum)
            grad_norm_avg = grad_norm_sum/p_sum
            grad_norm_avg_running = grad_norm_avg +0.9*0.5*grad_norm_avg_running
            net_eval = copy.deepcopy(net_glob)
            net_eval.load_state_dict(grad_avg)
            grad_avg_norm = torch.linalg.norm(parameters_to_vector(net_eval.parameters()))**2
            
            eta_g = (0.5*grad_norm_avg/(grad_avg_norm + m*0.1))
            eta_g = max(1,eta_g)

            w_vec_prev = w_vec_estimate

            w_vev_prev= copy.deepcopy(net_glob.state_dict())
            
            w_vec_estimate = Add(net_glob.state_dict(), Mul(grad_avg,eta_g))



            if(iter>0):
                w_vec_avg = Div(Add(w_vec_estimate,w_vec_prev),2)
            else:
                w_vec_avg = w_vec_estimate


        # copy weight to net_glob
        # net_glob.load_state_dict(w_glob)

        # w_old = copy.deepcopy(w_glob)
        # vector_to_parameters(w_vec_estimate,net_glob.parameters())
        net_glob.load_state_dict(w_vec_estimate)

        net_eval = copy.deepcopy(net_glob)
        net_eval.load_state_dict(w_vec_avg)
        # vector_to_parameters(w_vec_avg, net_eval.parameters())
        # vector_to_parameters(w_vec_avg, net_eval.parameters())

        if iter % 10 == 9:
            item_acc,item_loss = test_img(net_eval, dataset_test, args)
            acc.append(item_acc)
            loss.append(item_loss)

            print("Testing accuracy: {:.2f}".format(item_acc))
            print("Testing loss: {:.2f}".format(item_loss))

    save_result(acc, 'test_acc', args)
    save_result(loss, 'test_loss', args)


class LocalUpdate_FedExP(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.verbose = verbose

    def train(self, net):

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
                log_probs = net(images)['output']
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                Predict_loss += loss.item()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net.state_dict(), net.parameters()


