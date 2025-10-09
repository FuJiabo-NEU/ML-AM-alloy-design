# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from scipy import sparse
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import geatpy as ea  # 导入geatpy库
import time
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pandas import Series, DataFrame
import sklearn.datasets as datasets
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec
import numpy as np
import time
import pandas as pd
import random
from scipy.optimize import curve_fit
from sklearn.model_selection import GridSearchCV
import joblib

# 导入数据
start1 = time.perf_counter()
# 读取CSV数据
df = pd.read_excel("AM_Data.xlsx")
feature_name = [column for column in df][1:]
sample_data = df.values[:, :]
Data_feature = sample_data[:, 1:-1]
Data_target = sample_data[:, -1]
print('Data_feature = ', Data_feature.tolist())
print('Data_target = ', Data_target.tolist())
start_time1 = time.time()

# 对特征数据标准化
ss_feature = preprocessing.StandardScaler()
Data_feature_nor = ss_feature.fit_transform(Data_feature)

# 邻接矩阵
neighbor_matrix = []
csv_file1 = csv.reader(open('data_matrix.csv',encoding='utf-8-sig'))
for content in csv_file1:
    content = list(map(float, content))
    if len(content) != 0:
        neighbor_matrix.append(content)

print('neighbor_matrix=', neighbor_matrix)
neighbor_matrix = np.array(neighbor_matrix)


def manipulate_feature(feature, max_node, features):
    feature = feature.reshape(-1, 1)
    feature[:, [0]] = (feature[:, [0]])
    result = np.zeros((max_node, features))
    result[:feature.shape[0], :feature.shape[1]] = feature
    feature = result
    feature = sparse.csr_matrix(feature)

    return feature

def normalize_adj(neighbor, max_node, feature):
    np.fill_diagonal(neighbor, 1)
    neighbor = sparse.csr_matrix(neighbor)
    return neighbor

def normalize_t_label(label_matrix):
    label_mean = np.mean(label_matrix)
    label_std = np.std(label_matrix)
    label_matrix = (label_matrix - label_mean) / label_std

    norm = np.array([label_mean, label_std])
    np.savez_compressed('norm.npz', norm=norm)

    return label_matrix


def macro_avg_err(Y_prime, Y):
    if type(Y_prime) is np.ndarray:
        return np.sum(np.abs(Y - Y_prime)) / np.sum(np.abs(Y))
    return torch.sum(torch.abs(Y - Y_prime)) / torch.sum(torch.abs(Y))

def tensor_to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x.float())

def variable_to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    return x

def cnn_feature(feature):
    feature = feature.reshape(-1, 1)
    feature = sparse.csr_matrix(feature)

    return feature

# 生成图结构
class GraphTrainSet(Dataset):
    def __init__(self, train_x, train_y):
        max_node = 23
        num_features = 1
        neighbor_matrix_nor = neighbor_matrix

        for i in range(len(train_x)):
            train_feature = manipulate_feature(train_x[i], max_node, num_features)
            train_adj_matrix = normalize_adj(neighbor_matrix_nor, max_node, train_x[i])
            train_label = [train_y[i]]

            train_multiple_neighbor = [train_adj_matrix]
            train_multiple_feature = [train_feature]

            if i == 0:
                train_adjacency_matrix, train_node_attr_matrix, train_label_matrix = train_multiple_neighbor, train_multiple_feature, train_label
            else:
                train_adjacency_matrix, train_node_attr_matrix, train_label_matrix = np.concatenate((train_adjacency_matrix, train_multiple_neighbor)), \
                                                                                     np.concatenate((train_node_attr_matrix,train_multiple_feature)), \
                                                                                     np.concatenate((train_label_matrix,train_label))
        train_label_matrix = train_label_matrix.reshape(len(train_x), 1)
        self.train_adjacency_matrix = np.array(train_adjacency_matrix)
        self.train_node_attr_matrix = np.array(train_node_attr_matrix)
        self.train_label_matrix = np.array(train_label_matrix)


    def __len__(self):
        return len(self.train_adjacency_matrix)

    def __getitem__(self, idx):
        train_adjacency_matrix = self.train_adjacency_matrix[idx].todense()
        train_node_attr_matrix = self.train_node_attr_matrix[idx].todense()
        train_label_matrix = self.train_label_matrix[idx]
        # print('--------------------')
        train_adjacency_matrix = torch.from_numpy(train_adjacency_matrix)
        train_node_attr_matrix = torch.from_numpy(train_node_attr_matrix)
        train_label_matrix = torch.from_numpy(train_label_matrix)

        return train_adjacency_matrix, train_node_attr_matrix, train_label_matrix

class GraphTestSet(Dataset):
    def __init__(self, test_x, test_y):
        max_node = 23
        num_features = 1
        neighbor_matrix_nor = neighbor_matrix

        for i in range(len(test_x)):
            test_feature = manipulate_feature(test_x[i], max_node, num_features)
            test_adj_matrix = normalize_adj(neighbor_matrix_nor, max_node, test_x[i])
            test_label = [test_y[i]]

            test_multiple_neighbor = [test_adj_matrix]
            test_multiple_feature = [test_feature]

            if i == 0:
                test_adjacency_matrix, test_node_attr_matrix, test_label_matrix = test_multiple_neighbor, test_multiple_feature, test_label
            else:
                test_adjacency_matrix, test_node_attr_matrix, test_label_matrix = np.concatenate((test_adjacency_matrix, test_multiple_neighbor)), \
                                                                                     np.concatenate((test_node_attr_matrix,test_multiple_feature)), \
                                                                                     np.concatenate((test_label_matrix,test_label))
        test_label_matrix = test_label_matrix.reshape(len(test_x), 1)
        self.test_adjacency_matrix = np.array(test_adjacency_matrix)
        self.test_node_attr_matrix = np.array(test_node_attr_matrix)
        self.test_label_matrix = np.array(test_label_matrix)


    def __len__(self):
        return len(self.test_adjacency_matrix)

    def __getitem__(self, idx):
        test_adjacency_matrix = self.test_adjacency_matrix[idx].todense()
        test_node_attr_matrix = self.test_node_attr_matrix[idx].todense()
        test_label_matrix = self.test_label_matrix[idx]
        # print('--------------------')
        test_adjacency_matrix = torch.from_numpy(test_adjacency_matrix)
        test_node_attr_matrix = torch.from_numpy(test_node_attr_matrix)
        test_label_matrix = torch.from_numpy(test_label_matrix)

        return test_adjacency_matrix, test_node_attr_matrix, test_label_matrix

class GAT_Set(Dataset):
    def __init__(self, train_x):
        max_node = 23
        num_features = 1
        neighbor_matrix_nor = neighbor_matrix

        for i in range(len(train_x)):
            train_feature = manipulate_feature(train_x[i], max_node, num_features)
            train_adj_matrix = normalize_adj(neighbor_matrix_nor, max_node, train_x[i])

            train_multiple_neighbor = [train_adj_matrix]
            train_multiple_feature = [train_feature]

            if i == 0:
                train_adjacency_matrix, train_node_attr_matrix = train_multiple_neighbor, train_multiple_feature
            else:
                train_adjacency_matrix, train_node_attr_matrix = \
                    np.concatenate((train_adjacency_matrix, train_multiple_neighbor)), np.concatenate((train_node_attr_matrix, train_multiple_feature))

        self.train_adjacency_matrix = np.array(train_adjacency_matrix)
        self.train_node_attr_matrix = np.array(train_node_attr_matrix)


    def __len__(self):
        return len(self.train_adjacency_matrix)

    def __getitem__(self, idx):
        train_adjacency_matrix = self.train_adjacency_matrix[idx].todense()
        train_node_attr_matrix = self.train_node_attr_matrix[idx].todense()
        train_adjacency_matrix = torch.from_numpy(train_adjacency_matrix)
        train_node_attr_matrix = torch.from_numpy(train_node_attr_matrix)

        return train_adjacency_matrix, train_node_attr_matrix

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, relu, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.relu = relu
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.act = nn.LeakyReLU(relu)
        self.dropout = 0.0
    def forward(self, inp, adj):

        h = torch.matmul(inp, self.W)
        N = h.size()[1]
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1,N,N,2 * self.out_features)
        e = self.act(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -1e20 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.gelu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GATModel_1layer(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, n_heads, relu):

        super(GATModel_1layer, self).__init__()
        self.n_class = n_class
        self.dropout = 0.1
        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, relu, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        if self.n_class != 0 :
            self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, relu, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        if self.n_class != 0 :
            x = F.gelu(self.out_att(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        return x

class GAT(nn.Module):
    def __init__(self, p, logvar, relu):
        super(GAT, self).__init__()
        self.p = p
        self.logvar = logvar

        self.graph_modules = nn.Sequential(OrderedDict([
            ('GAT_layer_0', GATModel_1layer(n_feat=1, n_hid=2, n_class=0, n_heads=2, relu=relu)),
        ]))

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=23 * 2 * 2, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.69)
        )

        self.pre = nn.Sequential(
            nn.Linear(in_features=128, out_features=16),
            nn.GELU(),
        )

        self.var = nn.Sequential(
        )

        self.predict = nn.Sequential(
            nn.Linear(in_features=16, out_features=1),
            nn.ReLU(),
        )  # output layer

        self.get_var = nn.Sequential(
            nn.Linear(in_features=16, out_features=1),
        )  # output layer

    def forward(self, node_attr_matrix, adjacency_matrix):
        adjacency_matrix = adjacency_matrix.float()
        train_x = node_attr_matrix.float()
        x = train_x[:, :23, :]

        for (name, module) in self.graph_modules.named_children():
            if 'GAT_layer' in name:
                x = module(x, adj=adjacency_matrix)
            else:
                x = module(x)

        x = x.view(x.size()[0], -1)
        xc = self.fc1(x)
        # get y_mean
        x = self.pre(xc)
        y = self.predict(x)

        # get var
        xv = self.var(x)
        if self.var:
            var = self.get_var(xv)
        else:
            var = torch.zeros(y.size())
        return y, var, xc

class Statistical_loss(nn.Module):
    def __init__(self):
        super(Statistical_loss, self).__init__()

    def forward(self, gt, pred_y, logvar):
        gt = gt.cuda()
        pred_y = pred_y.cuda()
        logvar = logvar.cuda()
        loss = torch.mean(0.5*(torch.exp((-1)*logvar)) * (gt - pred_y)**2 + 0.5*logvar)
        return loss

def train(model, train_data_loader, epochs, optimizer, criterion, scheduler, num_ens=1, beta_type=0.1,):
    print(' ')
    print("*** Training started! ***")
    print(' ')
    # max_r2 = 0
    # early_iter = 0
    for epoch in range(epochs):
        model.train()
        total_macro_loss = []
        training_loss = 0.0
        kl_list = []

        for batch_id, (adjacency_matrix, node_attr_matrix, label_matrix) in enumerate(train_data_loader):
            adjacency_matrix = tensor_to_variable(adjacency_matrix)
            node_attr_matrix = tensor_to_variable(node_attr_matrix)

            label_matrix = tensor_to_variable(label_matrix)

            optimizer.zero_grad()
            y_pre, log_var = model(adjacency_matrix=adjacency_matrix, node_attr_matrix=node_attr_matrix)
            loss = criterion(label_matrix, y_pre, log_var)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            print('Epoch: {} \tTraining Loss: {:.4f} \t'.format(epoch, loss.item()))

        if epoch >= 1000:
            scheduler.step()

    return


def get_MC_Predictions(network, data_loader, dropout=True, mc_times=64):

    label_list, pre_list, ep_list, al_list, un_list = [], [], [], [], []

    for batch_id, (adjacency_matrix, node_attr_matrix, label_matrix) in enumerate(data_loader):
        adjacency_matrix = tensor_to_variable(adjacency_matrix)

        inputs = tensor_to_variable(node_attr_matrix)
        labels = tensor_to_variable(label_matrix)

        pred_v = np.zeros((len(labels), 1, mc_times))
        a_u = np.zeros((len(labels), 1, mc_times))
        for t in range(mc_times):
            if dropout:
                prediction, var = network(adjacency_matrix=adjacency_matrix, node_attr_matrix=inputs)
            else:
                network.eval()
                prediction, var = network(adjacency_matrix=adjacency_matrix, node_attr_matrix=inputs)

            prediction = prediction.detach().cpu().numpy()
            pred_v[:, :, t] = prediction
            var = var.detach().cpu().numpy()
            a_u[:, :, t] = var

        a_u = np.sqrt(np.exp(np.mean(a_u, axis=2)))
        pred_mean = np.mean(pred_v, axis=2)
        e_u = np.sqrt(np.var(pred_v, axis=2))
        un = a_u + e_u

        label_list.extend(variable_to_numpy(labels))
        pre_list.extend(pred_mean)
        al_list.extend(a_u)
        ep_list.extend(e_u)
        un_list.extend(un)

    label_list = np.array(label_list)
    pre_list = np.array(pre_list)
    al_list = np.array(al_list)
    ep_list = np.array(ep_list)
    un_list = np.array(un_list)

    return label_list, pre_list, ep_list, al_list, un_list

