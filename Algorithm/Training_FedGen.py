#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.generator import Generator
from models.Update import LocalUpdate_FedGen,DatasetSplit
from models.Fed import Aggregation
from models.test import test_img
from utils.utils import save_result
from utils.model_config import FedGenRUNCONFIGS


MIN_SAMPLES_PER_LABEL=1

def init_configs(args):


    RUNCONFIGS = FedGenRUNCONFIGS
    #### used for ensemble learning ####
    dataset_name = args.dataset
    args.ensemble_lr = RUNCONFIGS[dataset_name].get('ensemble_lr', 1e-4)
    args.ensemble_batch_size = RUNCONFIGS[dataset_name].get('ensemble_batch_size', 128)
    args.ensemble_epochs= RUNCONFIGS[dataset_name]['ensemble_epochs']
    args.num_pretrain_iters = RUNCONFIGS[dataset_name]['num_pretrain_iters']
    args.temperature = RUNCONFIGS[dataset_name].get('temperature', 1)
    args.ensemble_alpha = RUNCONFIGS[dataset_name].get('ensemble_alpha', 1)
    args.ensemble_beta = RUNCONFIGS[dataset_name].get('ensemble_beta', 0)
    args.ensemble_eta = RUNCONFIGS[dataset_name].get('ensemble_eta', 1)
    args.weight_decay = RUNCONFIGS[dataset_name].get('weight_decay', 0)
    args.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
    args.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']
    args.ensemble_train_loss = []
    args.n_teacher_iters = 5
    args.n_student_iters = 1


def read_user_data(args, dataset_train, dict_users):

    label_counts_users = []

    for idx in range(len(dict_users)):
        data_loader = DataLoader(DatasetSplit(dataset_train,dict_users[idx]),len(dict_users[idx]))
        for _,y in data_loader:
            unique_y, counts = torch.unique(y, return_counts=True)
        label_counts = [0 for i in range(args.num_classes)]
        for label, count in zip(unique_y, counts):
            label_counts[int(label)] += count
        label_counts_users.append(label_counts)

    return label_counts_users

def FedGen(args, net_glob, dataset_train, dataset_test, dict_users):

    init_configs(args)

    net_glob.train()

    generative_model = Generator(args.dataset, args.model, embedding=False, latent_layer_idx=-1)

    label_counts = read_user_data(args, dataset_train, dict_users)

    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    acc = []

    for iter in range(args.epochs):

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        w_locals = []
        lens = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        user_models = []

        for idx in idxs_users:
            local = LocalUpdate_FedGen(args=args, generative_model=generative_model, dataset=dataset_train, idxs=dict_users[idx], regularization=iter!=0)
            user_model = local.train(net=copy.deepcopy(net_glob).to(args.device))
            user_models.append(user_model)
            w_locals.append(copy.deepcopy(user_model.state_dict()))
            lens.append(len(dict_users[idx]))
        net_glob.to('cpu')
        train_generator(
            args,
            net_glob,
            generative_model,
            user_models,
            idxs_users,
            label_counts,
            args.bs,
            epoches=args.ensemble_epochs // args.n_teacher_iters,
            latent_layer_idx = -1,
            verbose=True
        )
        net_glob.to(args.device)
        # update global weights
        w_glob = Aggregation(w_locals, lens)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        if iter % 10 == 9:
            acc.append(test(net_glob, dataset_test, args))

    save_result(acc, 'test_acc', args)

def get_label_weights(args, users, label_counts):
    label_weights = []
    qualified_labels = []
    for label in range(args.num_classes):
        weights = []
        for user in users:
            weights.append(label_counts[user][label])
        if np.max(weights) > MIN_SAMPLES_PER_LABEL:
            qualified_labels.append(label)
        # uniform
        label_weights.append( np.array(weights) / np.sum(weights) )
    label_weights = np.array(label_weights).reshape((args.num_classes, -1))
    return label_weights, qualified_labels

def train_generator(args, net_glob, generative_model, models, users, label_counts, batch_size, epoches=1, latent_layer_idx=-1, verbose=False):
    """
    Learn a generator that find a consensus latent representation z, given a label 'y'.
    :param batch_size:
    :param epoches:
    :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
    :param verbose: print loss information.
    :return: Do not return anything.
    """

    label_weights, qualified_labels = get_label_weights(args, users, label_counts)
    TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0

    generative_optimizer = torch.optim.Adam(params=generative_model.parameters(),lr=args.ensemble_lr,weight_decay=args.weight_decay)

    for i in range(epoches):

        generative_model.train()
        net_glob.eval()

        for i in range(args.n_teacher_iters):
            generative_optimizer.zero_grad()

            y = np.random.choice(qualified_labels, batch_size)
            y_input = torch.LongTensor(y)
            ## feed to generator
            gen_result = generative_model(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
            # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
            gen_output, eps = gen_result['output'], gen_result['eps']
            ##### get losses ####
            # decoded = self.generative_regularizer(gen_output)
            # regularization_loss = beta * self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
            diversity_loss = generative_model.diversity_loss(eps, gen_output)  # encourage different outputs

            ######### get teacher loss ############
            teacher_loss = 0
            teacher_logit = 0
            for user_idx, user_model in enumerate(models):
                weight = label_weights[y][:, user_idx].reshape(-1, 1)
                expand_weight = np.tile(weight, (1, args.num_classes))
                user_result_given_gen = user_model(gen_output, start_layer_idx=latent_layer_idx)
                user_output_logp_ = user_result_given_gen['output']
                teacher_loss_=torch.mean( \
                    generative_model.crossentropy_loss(user_output_logp_, y_input) * \
                    torch.tensor(weight, dtype=torch.float32))
                teacher_loss += teacher_loss_
                teacher_logit += user_result_given_gen['output'] * torch.tensor(expand_weight, dtype=torch.float32)

            ######### get student loss ############
            student_output = net_glob(gen_output, start_layer_idx=latent_layer_idx)
            student_loss = F.kl_div(F.log_softmax(student_output['output'], dim=1), F.softmax(teacher_logit, dim=1))
            if args.ensemble_beta > 0:
                loss = args.ensemble_alpha * teacher_loss - args.ensemble_beta * student_loss + args.ensemble_eta * diversity_loss
            else:
                loss = args.ensemble_alpha * teacher_loss + args.ensemble_eta * diversity_loss
            loss.backward()
            generative_optimizer.step()
            TEACHER_LOSS += args.ensemble_alpha * teacher_loss.item()#(torch.mean(TEACHER_LOSS.double())).item()
            STUDENT_LOSS += args.ensemble_beta * student_loss.item()#(torch.mean(student_loss.double())).item()
            DIVERSITY_LOSS += args.ensemble_eta * diversity_loss.item()#(torch.mean(diversity_loss.double())).item()

    TEACHER_LOSS = TEACHER_LOSS / (args.n_teacher_iters * epoches)
    STUDENT_LOSS = STUDENT_LOSS / (args.n_teacher_iters * epoches)
    DIVERSITY_LOSS = DIVERSITY_LOSS / (args.n_teacher_iters * epoches)
    info="Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
        format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
    if verbose:
        print(info)

def test(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()