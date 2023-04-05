""" Training and testing of the model
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from utils import (cal_adj_mat_parameter, gen_adj_mat_tensor,
                   gen_test_adj_mat_tensor)
from models import cotraining
cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    print(num_view)
    labels_tr = np.loadtxt(os.path.join(
        data_folder, "labels_tr.csv"), delimiter=',')
    print(labels_tr)
    labels_te = np.loadtxt(os.path.join(
        data_folder, "labels_te.csv"), delimiter=',')
    print(labels_te)
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []

    for i in view_list:
        data_te_list.append(np.genfromtxt(os.path.join(
            data_folder, str(i) + "_te.csv"), delimiter=','))
        data_tr_list.append(np.genfromtxt(os.path.join(
            data_folder, str(i) + "_tr.csv"), delimiter=','))
    print(data_te_list)

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate(
            (data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                        data_tensor_list[i][idx_dict["te"]].clone()), 0))
    labels = np.concatenate((labels_tr, labels_te))
    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine"  # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(
            adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(
            data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(
            data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))

    return adj_train_list, adj_test_list

def train_epoch(data_list, adj_list, label, onehot_labels_tr_tensor, sample_weight, model_dict, optim_dict, train_VCDN=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)
    is_cotrain = True
    if is_cotrain:
        cotraining(optim_dict, data_list, model_dict, criterion,
                   sample_weight, label, adj_list, loss_dict)
    else:
        for i in range(num_view):
            optim_dict["C{:}".format(i+1)].zero_grad()
            ci_loss = 0
            ci = model_dict["C{:}".format(
                i+1)](model_dict["E{:}".format(i+1)](data_list[i], adj_list[i]))
            ci_loss = torch.mean(
                torch.mul(criterion(ci, label), sample_weight))
            ci_loss.backward()
            optim_dict["C{:}".format(i+1)].step()
            loss_dict["C{:}".format(
                i+1)] = ci_loss.detach().cpu().numpy().item()

    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["C{:}".format(
                i+1)](model_dict["E{:}".format(i+1)](data_list[i], adj_list[i])))
        c = model_dict["C"](ci_list)
        c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
    return loss_dict


def test_epoch(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(
            i+1)](model_dict["E{:}".format(i+1)](data_list[i], adj_list[i])))
    if num_view >= 2:
        c = model_dict["C"](ci_list)
    else:
        c = ci_list[0]
    c = c[te_idx, :]
    prob = F.softmax(c, dim=1).data.cpu().numpy()

    return prob
