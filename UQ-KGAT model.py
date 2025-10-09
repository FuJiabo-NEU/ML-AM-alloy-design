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
sample_data = df.values[:, :]  # 多尺度数据集
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

# 特征矩阵转换成稀疏矩阵：坐标 + 数值
def manipulate_feature(feature, max_node, features):

    feature = feature.reshape(-1, 1) # 转换成1列 (32, 1)
    feature[:, [0]] = (feature[:, [0]]) #
    # 匹配特征features的最大维度
    result = np.zeros((max_node, features)) # 创建(32, 1)的零矩阵
    result[:feature.shape[0], :feature.shape[1]] = feature
    feature = result
    # 将特征矩阵转换成稀疏矩阵 形式: 坐标 + 值
    feature = sparse.csr_matrix(feature)

    return feature

# 邻接矩阵对称归一化并转换成稀疏矩阵：D' * A' * D'
def normalize_adj(neighbor, max_node, feature):

    np.fill_diagonal(neighbor, 1)
    neighbor = sparse.csr_matrix(neighbor)
    # print(neighbor)
    return neighbor

# 标签矩阵Z-score标准化并返回均值、偏差
def normalize_t_label(label_matrix):
    label_mean = np.mean(label_matrix)
    label_std = np.std(label_matrix)
    label_matrix = (label_matrix - label_mean) / label_std # Z-score标准化

    # save the mean and standard deviation of label
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
    #return Variable(x)


def variable_to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    return x


# 特征矩阵转换成稀疏矩阵：坐标 + 数值
def cnn_feature(feature):

    feature = feature.reshape(-1, 1) # 转换成1列 (32, 1)
    feature = sparse.csr_matrix(feature)

    return feature

# 生成图结构
class GraphTrainSet(Dataset):
    def __init__(self, train_x, train_y):
        max_node = 23
        num_features = 1
        neighbor_matrix_nor = neighbor_matrix

        # 划分后的数据生成图结构
        for i in range(len(train_x)):
            # feature data manipulation
            train_feature = manipulate_feature(train_x[i], max_node, num_features)
            # normalize the adjacency matrix
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
        # train_label_matrix = normalize_t_label(train_label_matrix)
        self.train_adjacency_matrix = np.array(train_adjacency_matrix)
        self.train_node_attr_matrix = np.array(train_node_attr_matrix)  # 节点编号+特征值
        self.train_label_matrix = np.array(train_label_matrix)
        # print('Training Data:')
        # print('train_adjacency matrix:\t', self.train_adjacency_matrix.shape)
        # print('train_node attribute matrix:\t', self.train_node_attr_matrix.shape)
        # print('train_label name:\t\t', self.train_label_matrix.shape)

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
        # neighbor_matrix = np.ones((32, 32))
        # np.fill_diagonal(neighbor_matrix, 1)
        # 划分后的数据生成图结构
        for i in range(len(test_x)):
            # feature data manipulation
            test_feature = manipulate_feature(test_x[i], max_node, num_features)
            # normalize the adjacency matrix
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
        self.test_node_attr_matrix = np.array(test_node_attr_matrix)  # 节点编号+特征值
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

        # 划分后的数据生成图结构
        for i in range(len(train_x)):
            # feature data manipulation
            train_feature = manipulate_feature(train_x[i], max_node, num_features)
            # normalize the adjacency matrix
            train_adj_matrix = normalize_adj(neighbor_matrix_nor, max_node, train_x[i])

            train_multiple_neighbor = [train_adj_matrix]
            train_multiple_feature = [train_feature]

            if i == 0:
                train_adjacency_matrix, train_node_attr_matrix = train_multiple_neighbor, train_multiple_feature
            else:
                train_adjacency_matrix, train_node_attr_matrix = \
                    np.concatenate((train_adjacency_matrix, train_multiple_neighbor)), np.concatenate((train_node_attr_matrix, train_multiple_feature))
        # train_label_matrix = normalize_t_label(train_label_matrix)
        self.train_adjacency_matrix = np.array(train_adjacency_matrix)
        self.train_node_attr_matrix = np.array(train_node_attr_matrix)  # 节点编号+特征值


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


# GCN搭建
class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, relu, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征数
        self.out_features = out_features  # 节点表示向量的输出特征数
        self.relu = relu
        self.concat = concat  # 如果为true, 再进行elu激活

        # GAT
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化 1.414

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化

        # 定义leakyrelu激活函数
        self.act = nn.LeakyReLU(relu)
        self.dropout = 0.0
    def forward(self, inp, adj):

        h = torch.matmul(inp, self.W)  # [B, N, out_features]
        N = h.size()[1]  # N 图的节点数
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1,N,N,2 * self.out_features)
        e = self.act(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -1e20 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [B, N, N]
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)  # [B, N, N].[B, N, out_features] => [B, N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            # return F.leaky_relu(h_prime, negative_slope=0.01)
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
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        if self.n_class != 0 :
            self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, relu, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # 将每个head得到的表示进行拼接
        # x = F.dropout(x, self.dropout, training=self.training)
        if self.n_class != 0 :
            x = F.gelu(self.out_att(x, adj)) # 输出并激活
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


def GAT_PRE(network, data_loader, dropout=True, mc_times=64):

    pre_list, ep_list, al_list, un_list = [], [], [], []
    for batch_id, (adjacency_matrix, node_attr_matrix) in enumerate(data_loader):
        adjacency_matrix = tensor_to_variable(adjacency_matrix)
        inputs = tensor_to_variable(node_attr_matrix)

        pred_v = np.zeros((len(inputs), 1, mc_times))
        a_u = np.zeros((len(inputs), 1, mc_times))
        for t in range(mc_times):
            if dropout:
                prediction, var, _ = network(adjacency_matrix=adjacency_matrix, node_attr_matrix=inputs)
            else:
                network.eval()
                prediction, var, _ = network(adjacency_matrix=adjacency_matrix, node_attr_matrix=inputs)

            prediction = prediction.detach().cpu().numpy()
            pred_v[:, :, t] = prediction
            var = var.detach().cpu().numpy()
            a_u[:, :, t] = var

        a_u = np.sqrt(np.exp(np.mean(a_u, axis=2)))
        pred_mean = np.mean(pred_v, axis=2)
        e_u = np.sqrt(np.var(pred_v, axis=2))
        un = a_u + e_u

        pre_list.extend(pred_mean)
        al_list.extend(a_u)
        ep_list.extend(e_u)
        un_list.extend(un)

    pre_list = np.array(pre_list)
    al_list = np.array(al_list)
    ep_list = np.array(ep_list)
    un_list = np.array(un_list)

    return pre_list, ep_list, al_list, un_list

