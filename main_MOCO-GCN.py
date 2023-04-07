import os
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,)
from sklearn.model_selection import KFold
from models import init_model_dict, init_optim
from train_test import (gen_trte_adj_mat, test_epoch,
                        train_epoch)
from utils import (cal_sample_weight,
                    one_hot_tensor)
import random
cuda = True if torch.cuda.is_available() else False
import argparse

#parser argument
parser = argparse.ArgumentParser(description='The data folder: X.csv is a microbiome and exposome abundance matrix, sample as rows, microbiome and exposome as columns, the last 6 or 23 rows are exposome data'
                                             'Y.csv is pancreatic cancer label')
parser.add_argument('-input', default=None, help='X.csv (The microbiome and exposome features as columns and samples as rows, the last 6 or 23 rows are exposome data'
                                                 'Y.csv (pancreatic cancer label)')

args = parser.parse_args()


if __name__ == '__main__':
    def seed_torch(seed=0):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    seed_torch()
    data_folder = args.input
    num_epoch_pretrain = 200
    num_epoch = 2000
    lr_e_pretrain = 1e-3
    lr_e = 5e-2
    lr_c = 1e-3
    num_class = 2
    num_view = 2
    dim_hvcdn = pow(num_class, num_view)
    adj_parameter = 8
    dim_he_list = [200, 200, 100, 100]
    kf = KFold(n_splits=5)
    X = np.loadtxt(os.path.join(data_folder, "X.csv"), delimiter=',')
    y = np.loadtxt(os.path.join(data_folder, "Y.csv"), delimiter=',')
    all_roc = []
    all_acc = []
    all_f1score = []
    for fold_num, (train, test) in enumerate(kf.split(X, y)):
        labels_tr = y[train]
        labels_te = y[test]
        labels_tr = labels_tr.astype(int)
        labels_te = labels_te.astype(int)
        data_tr_list = []
        data_te_list = []
        data_te_list.append(X[test][:, 0:125])
        data_te_list.append(X[test][:, 125:131])
        data_tr_list.append(X[train][:, 0:125])
        data_tr_list.append(X[train][:, 125:131])
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
        idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
        data_train_list = []
        data_all_list = []
        for i in range(len(data_tensor_list)):
            data_train_list.append(
                data_tensor_list[i][idx_dict["tr"]].clone())
            data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                            data_tensor_list[i][idx_dict["te"]].clone()), 0))
        labels_trte = np.concatenate((labels_tr, labels_te))
        data_tr_list = data_train_list
        data_trte_list = data_all_list
        trte_idx = idx_dict
        labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
        onehot_labels_tr_tensor = one_hot_tensor(
            labels_tr_tensor, num_class)
        sample_weight_tr = cal_sample_weight(
            labels_trte[trte_idx["tr"]], num_class)
        sample_weight_tr = torch.FloatTensor(sample_weight_tr)
        if cuda:
            labels_tr_tensor = labels_tr_tensor.cuda()
            onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
            sample_weight_tr = sample_weight_tr.cuda()
        adj_tr_list, adj_te_list = gen_trte_adj_mat(
            data_tr_list, data_trte_list, trte_idx, adj_parameter)
        dim_list = [x.shape[1] for x in data_tr_list]
        model_dict = init_model_dict(
            num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
        roc = []
        acc = []
        f1score = []
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()

        print("\nPretrain GCNs...")
        optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
        for epoch in range(num_epoch_pretrain):
            train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                        onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)
        print("\nTraining...")
        optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
        for epoch in range(num_epoch + 1):
            train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                        onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)

            te_prob = test_epoch(
                data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            roc.append(roc_auc_score(
                labels_trte[trte_idx["te"]], te_prob[:, 1]))
            acc.append(accuracy_score(
                labels_trte[trte_idx["te"]], te_prob.argmax(1)))
            f1score.append(f1_score(
                labels_trte[trte_idx["te"]], te_prob.argmax(1)))
        index = acc.index(max(acc))


        print("\n fold: {:.2f}".format(fold_num))
        print("\n ROC: {:.2f}".format(roc[index]))
        all_roc.append(roc[index])
        all_acc.append(acc[index])
        all_f1score.append(f1score[index])

    mean_roc = np.mean(all_roc)
    mean_acc = np.mean(all_acc)
    mean_f1score = np.mean(all_f1score)
    std_roc = np.std(all_roc, ddof=1)
    std_acc = np.std(all_acc, ddof=1)
    std_f1score = np.std(all_f1score, ddof=1)

    print("\nMean ROC: {:.2f}".format(mean_roc))
    print("\nMean ACC: {:.2f}".format(mean_acc))
    print("\nMean F1score: {:.2f}".format(mean_f1score))
    print("std roc: {:.2f}".format(std_roc))
    print("std acc: {:.2f}".format(std_acc))
    print("std F1score: {:.2f}".format(std_f1score))





