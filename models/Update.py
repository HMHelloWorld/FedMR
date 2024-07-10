#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import random
from optimizer.Adabelief import AdaBelief


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate_FedAvg(object):
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

        return net.state_dict()

class LocalUpdate_ClientSampling(object):
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

        return net

class LocalUpdate_FedProx(object):
    def __init__(self, args, glob_model, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.glob_model = glob_model
        self.prox_alpha = args.prox_alpha
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
        Penalize_loss = 0

        global_weight_collector = list(self.glob_model.parameters())

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)['output']
                predictive_loss = self.loss_func(log_probs, labels)

                # for fedprox
                fed_prox_reg = 0.0
                # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += ((self.prox_alpha / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)

                loss = predictive_loss + fed_prox_reg
                Predict_loss += predictive_loss.item()
                Penalize_loss += fed_prox_reg.item()

                loss.backward()
                optimizer.step()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            info += ', Penalize loss={:.4f}'.format(Penalize_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net.state_dict()


class LocalUpdate_Scaffold(object):

    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []

        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        if indd is not None:
            self.indd = indd
        else:
            self.indd = None

    def train(self, net, c_list={}, idx=-1):
        net.train()

        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                        weight_decay=1e-5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        num_updates = 0
        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(self.args.device)

                log_probs = net(images)['output']
                loss_fi = self.loss_func(log_probs, labels)

                local_par_list = None
                dif = None
                for param in net.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                for k in c_list[idx].keys():
                    if not isinstance(dif, torch.Tensor):
                        dif = (-c_list[idx][k] + c_list[-1][k]).reshape(-1)
                    else:
                        dif = torch.cat((dif, (-c_list[idx][k] + c_list[-1][k]).reshape(-1)), 0)
                loss_algo = torch.sum(local_par_list * dif)
                loss = loss_fi + loss_algo
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)
                optimizer.step()

                num_updates += 1

        return net.state_dict(), num_updates

class LocalUpdate_FedGKD(object):
    def __init__(self, args, glob_model, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.glob_model = glob_model.to(args.device)
        self.ensemble_alpha = args.ensemble_alpha
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
        Emsemble_loss = 0

        for iter in range(self.args.local_ep):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)['output']
                predictive_loss = self.loss_func(log_probs, labels)

                global_output_logp = self.glob_model(images)['output']

                user_latent_loss = self.ensemble_alpha * self.ensemble_loss(F.log_softmax(log_probs, dim=1),
                                                                        F.softmax(global_output_logp, dim=1))

                loss = predictive_loss + user_latent_loss
                Predict_loss += predictive_loss.item()
                Emsemble_loss += user_latent_loss.item()

                loss.backward()
                optimizer.step()

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            info += ', Emsemble loss={:.4f}'.format(Emsemble_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net.state_dict()

class LocalUpdate_Moon(object):
    def __init__(self, args, glob_model, old_models, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.glob_model = glob_model.to(args.device)
        self.old_models = old_models
        self.contrastive_alpha = args.contrastive_alpha
        self.temperature = args.temperature
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
        Contrastive_loss = 0

        for iter in range(self.args.local_ep):
            epoch_loss_collector = []
            epoch_loss1_collector = []
            epoch_loss2_collector = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                output = net(images)
                predictive_loss = self.loss_func(output['output'], labels)

                output_representation = output['representation']
                pos_representation = self.glob_model(images)['representation']
                posi = self.cos(output_representation, pos_representation)
                logits = posi.reshape(-1, 1)

                for previous_net in self.old_models:

                    neg_representation = previous_net(images)['representation']
                    nega = self.cos(output_representation, neg_representation)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                logits /= self.temperature
                labels = torch.zeros(images.size(0)).to(self.args.device).long()

                contrastive_loss = self.contrastive_alpha * self.loss_func(logits, labels)

                loss = predictive_loss + contrastive_loss
                Predict_loss += predictive_loss.item()
                Contrastive_loss += contrastive_loss.item()

                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())
                epoch_loss1_collector.append(predictive_loss.item())
                epoch_loss2_collector.append(contrastive_loss.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
            epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
            if self.verbose:
                print('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (iter, epoch_loss, epoch_loss1, epoch_loss2))

        if self.verbose:
            info = '\nUser predict Loss={:.4f}'.format(Predict_loss / (self.args.local_ep * len(self.ldr_train)))
            info += ', Contrastive loss={:.4f}'.format(Contrastive_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        return net.state_dict()

class LocalUpdate_FedGen(object):
    def __init__(self, args, generative_model, dataset=None, idxs=None, verbose=False, regularization=True):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ensemble_loss = nn.KLDivLoss(reduction='batchmean')
        self.crossentropy_loss = nn.CrossEntropyLoss(reduce=False)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.verbose = verbose
        self.generative_model = generative_model
        self.regularization = regularization
        self.generative_alpha = args.generative_alpha
        self.generative_beta = args.generative_beta
        self.latent_layer_idx = -1

    def train(self, net):

        net.train()
        self.generative_model.eval()

        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=self.args.lr)

        Predict_loss = 0
        Teacher_loss = 0
        Latent_loss = 0
        for iter in range(self.args.local_ep):

            for batch_idx, (images, y) in enumerate(self.ldr_train):
                images, y = images.to(self.args.device), y.to(self.args.device)
                net.zero_grad()
                user_output_logp = net(images)['output']
                predictive_loss = self.loss_func(user_output_logp, y)

                #### sample y and generate z
                if self.regularization:
                    ### get generator output(latent representation) of the same label
                    gen_output = self.generative_model(y, latent_layer_idx=self.latent_layer_idx)['output'].to(
                        self.args.device)
                    logit_given_gen = net(gen_output, start_layer_idx=self.latent_layer_idx)['output']
                    target_p = F.softmax(logit_given_gen, dim=1).clone().detach()
                    user_latent_loss = self.generative_beta * self.ensemble_loss(F.log_softmax(user_output_logp, dim=1),
                                                                            target_p)

                    sampled_y = np.random.choice(self.args.num_classes, self.args.bs)
                    sampled_y = torch.LongTensor(sampled_y).to(self.args.device)
                    gen_result = self.generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx)
                    gen_output = gen_result['output'].to(
                        self.args.device)  # latent representation when latent = True, x otherwise
                    user_output_logp = net(gen_output, start_layer_idx=self.latent_layer_idx)['output']
                    teacher_loss = self.generative_alpha * torch.mean(
                        self.crossentropy_loss(user_output_logp, sampled_y)
                    )
                    # this is to further balance oversampled down-sampled synthetic data
                    gen_ratio = self.args.bs / self.args.bs
                    loss = predictive_loss + gen_ratio * teacher_loss + user_latent_loss
                    Teacher_loss += teacher_loss.item()
                    Latent_loss += user_latent_loss.item()
                else:
                    #### get loss and perform optimization
                    loss = predictive_loss

                loss.backward()
                optimizer.step()

                Predict_loss += loss.item()

        if self.verbose:
            info = 'User predict Loss={:.4f} Teacher Loss={:.4f} Latent Loss={:.4f}'.format(
                Predict_loss / (self.args.local_ep * len(self.ldr_train)),
                Teacher_loss / (self.args.local_ep * len(self.ldr_train)),
                Latent_loss / (self.args.local_ep * len(self.ldr_train)))
            print(info)

        net.to('cpu')

        return net
    

class LocalUpdate_FedSA(object):
    def __init__(self, args, dataset=None, idxs=None, verbose=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.verbose = verbose

    def train(self, net, lr):

        net.train()
        # train and update
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum,
                                        weight_decay=1e-5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(net.parameters(), lr=lr)

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

        return net.state_dict()