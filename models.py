""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
           

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    

class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()

        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.gc4 = GraphConvolution(hgcn_dim[2], hgcn_dim[3])
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        x = F.leaky_relu(x, 0.25)
        return x

def cotraining(optim_dict, data_list, model_dict, criterion, sample_weight, label, adj_list, loss_dict):
    """
    Description:
    fits the classifiers on the partially labeled data, y.

    Parameters:
    X1 - array-like (n_samples, n_features_1): first set of features for samples
    X2 - array-like (n_samples, n_features_2): second set of features for samples
    y - array-like (n_samples): labels for samples, -1 indicates unlabeled
    p - (Optional) The number of positive examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)

	n - (Optional) The number of negative examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)

	k - (Optional) The number of iterations
		The default is 30 (from paper)

	u - (Optional) The size of the pool of unlabeled samples from which the classifier can choose
		Default - 75 (from paper)

    """
    num_view = len(data_list)
    # we need y to be a numpy array so we can do more complex slicing
    y = np.asarray(label.data.numpy().copy())
    for _ in range(len(label)//3):
        i = random.choice(range(len(label)))
        y[i] = -1
    p_ = -1
    n_ = -1
    k_ = 30
    u_ = 15
    # set the n and p parameters if we need to
    if p_ == -1 and n_ == -1:
        num_pos = sum(1 for y_i in y if y_i == 1)
        num_neg = sum(1 for y_i in y if y_i == 0)

        n_p_ratio = num_neg / float(num_pos)

        if n_p_ratio > 1:
            p_ = 1
            n_ = round(p_*n_p_ratio)

        else:
            n_ = 1
            p_ = round(n_/n_p_ratio)

    assert(p_ > 0 and n_ > 0 and k_ > 0 and u_ > 0)

    # the set of unlabeled samples
    U = [i for i, y_i in enumerate(y) if y_i == -1]

    # we randomize here, and then just take from the back so we don't have to sample every time
    random.shuffle(U)

    # this is U' in paper
    U_ = U[-min(len(U), u_):]

    # the samples that are initially labeled
    L = [i for i, y_i in enumerate(y) if y_i != -1]

    # remove the samples in U_ from U
    U = U[:-len(U_)]

    it = 0  # number of cotraining iterations we've done so far
    # loop until we have assigned labels to everything in U or we hit our iteration break condition
    while it != k_ and U:
        it += 1
        y_prob = []

        for i in range(num_view):
            optim_dict["C{:}".format(i+1)].zero_grad()
            ci_loss = 0
            ci = model_dict["C{:}".format(
                i+1)](model_dict["E{:}".format(i+1)](data_list[i], adj_list[i]))
            y_prob.append(ci)
            ci_loss = torch.mean(
                torch.mul(criterion(torch.index_select(ci, 0, torch.tensor(L)), label[L]), sample_weight[L]))
            ci_loss.backward()
            optim_dict["C{:}".format(i+1)].step()

        y1_prob, y2_prob = torch.index_select(y_prob[0], 0, torch.tensor(
            U_)), torch.index_select(y_prob[1], 0, torch.tensor(U_))

        n, p = [], []

        for i in (y1_prob[:, 0].argsort())[-n_:]:
            if y1_prob[i, 0] > 0.5:
                n.append(i)
        for i in (y1_prob[:, 1].argsort())[-p_:]:
            if y1_prob[i, 1] > 0.5:
                p.append(i)

        for i in (y2_prob[:, 0].argsort())[-n_:]:
            if y2_prob[i, 0] > 0.5:
                n.append(i)
        for i in (y2_prob[:, 1].argsort())[-p_:]:
            if y2_prob[i, 1] > 0.5:
                p.append(i)

        # label the samples and remove thes newly added samples from U_
        y[[U_[x] for x in p]] = 1
        y[[U_[x] for x in n]] = 0

        L.extend([U_[x] for x in p])
        L.extend([U_[x] for x in n])

        U_ = [elem for elem in U_ if not (elem in p or elem in n)]

        # add new elements to U_
        add_counter = 0  # number we have added from U to U_
        num_to_add = len(p) + len(n)
        while add_counter != num_to_add and U:
            add_counter += 1
            U_.append(U.pop())

        # TODO: Handle the case where the classifiers fail to agree on any of the samples (i.e. both n and p are empty)

    # let's fit our final model
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)].zero_grad()
        ci_loss = 0
        ci = model_dict["C{:}".format(
            i+1)](model_dict["E{:}".format(i+1)](data_list[i], adj_list[i]))
        ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i+1)].step()
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()



class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)
        
    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),(-1,pow(self.num_cls,2),1))
        for i in range(2,num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)),(-1,pow(self.num_cls,i+1),1))
        vcdn_feat = torch.reshape(x, (-1,pow(self.num_cls,num_view)))
        output = self.model(vcdn_feat)
        return output

    
def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dopout=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = GCN_E(dim_list[i], dim_he_list, gcn_dopout)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict


def init_optim(num_view, model_dict, lr_e=5e-4, lr_c=1e-3):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()), 
                lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict