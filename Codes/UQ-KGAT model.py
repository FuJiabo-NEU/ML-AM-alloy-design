# -*- coding: utf-8 -*-
"""
UQ-GAT-based Material Defect Prediction System
-------------------------------------------
集成图注意力网络(GAT)，用于材料缺陷预测
包含数据加载、模型构建、训练、预测
"""

# ==================== 核心库导入模块 ====================
# 功能：导入必要的深度学习、数据处理和优化算法库
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
import geatpy as ea  # 多目标优化算法库
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

# ==================== 数据加载与预处理模块 ====================
# 功能：加载Excel数据文件，提取特征和标签，并进行标准化处理
start1 = time.perf_counter()
# 读取CSV数据
df = pd.read_excel("AM_Data.xlsx")
feature_name = [column for column in df][1:]  # 提取特征名称
sample_data = df.values[:, :]  # 获取数据矩阵
Data_feature = sample_data[:, 1:-1]  # 特征数据
Data_target = sample_data[:, -1]  # 目标标签
print('Data_feature = ', Data_feature.tolist())
print('Data_target = ', Data_target.tolist())
start_time1 = time.time()

# 对特征数据标准化
ss_feature = preprocessing.StandardScaler()
Data_feature_nor = ss_feature.fit_transform(Data_feature)

# ==================== 图结构数据模块 ====================
# 功能：构建图邻接矩阵，定义图数据处理函数

# 邻接矩阵加载
neighbor_matrix = []
csv_file1 = csv.reader(open('data_matrix.csv',encoding='utf-8-sig'))
for content in csv_file1:
    content = list(map(float, content))
    if len(content) != 0:
        neighbor_matrix.append(content)

print('neighbor_matrix=', neighbor_matrix)
neighbor_matrix = np.array(neighbor_matrix)

# 功能：将特征矩阵转换为适合图神经网络的稀疏矩阵格式
def manipulate_feature(feature, max_node, features):
    """
    转换特征为图节点特征矩阵
    参数：
        feature: 原始特征数组
        max_node: 图中最大节点数
        features: 每个节点的特征维度
    返回：scipy稀疏矩阵格式的节点特征
    """
    feature = feature.reshape(-1, 1)
    feature[:, [0]] = (feature[:, [0]])
    result = np.zeros((max_node, features))
    result[:feature.shape[0], :feature.shape[1]] = feature
    feature = result
    feature = sparse.csr_matrix(feature)
    return feature

# 功能：邻接矩阵标准化处理
def normalize_adj(neighbor, max_node, feature):
    """
    邻接矩阵对称归一化
    参数：
        neighbor: 原始邻接矩阵
        max_node: 图中最大节点数
        feature: 特征矩阵（用于维度匹配）
    返回：稀疏矩阵格式的标准化邻接矩阵
    """
    np.fill_diagonal(neighbor, 1)  # 添加自连接
    neighbor = sparse.csr_matrix(neighbor)
    return neighbor

# 功能：标签矩阵标准化
def normalize_t_label(label_matrix):
    """
    标签Z-score标准化
    参数：
        label_matrix: 原始标签矩阵
    返回：标准化后的标签矩阵，保存均值和标准差
    """
    label_mean = np.mean(label_matrix)
    label_std = np.std(label_matrix)
    label_matrix = (label_matrix - label_mean) / label_std

    norm = np.array([label_mean, label_std])
    np.savez_compressed('norm.npz', norm=norm)

    return label_matrix

# 功能：计算平均绝对百分比误差
def macro_avg_err(Y_prime, Y):
    """
    计算预测值与真实值的平均绝对百分比误差
    参数：
        Y_prime: 预测值
        Y: 真实值
    返回：MAPE误差值
    """
    if type(Y_prime) is np.ndarray:
        return np.sum(np.abs(Y - Y_prime)) / np.sum(np.abs(Y))
    return torch.sum(torch.abs(Y - Y_prime)) / torch.sum(torch.abs(Y))

# 功能：张量与变量转换函数
def tensor_to_variable(x):
    """
    将PyTorch张量转换为Variable，支持GPU加速
    参数：
        x: 输入张量
    返回：Variable对象
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x.float())

def variable_to_numpy(x):
    """
    将Variable转换回numpy数组
    参数：
        x: 输入Variable
    返回：numpy数组
    """
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    return x

# ==================== 图数据集类模块 ====================
# 功能：定义用于GAT模型的PyTorch Dataset类

class GraphTrainSet(Dataset):
    """
    训练集图数据结构
    功能：将特征和标签转换为图结构数据，用于GAT模型训练
    """
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
        train_adjacency_matrix = torch.from_numpy(train_adjacency_matrix)
        train_node_attr_matrix = torch.from_numpy(train_node_attr_matrix)
        train_label_matrix = torch.from_numpy(train_label_matrix)
        return train_adjacency_matrix, train_node_attr_matrix, train_label_matrix

class GraphTestSet(Dataset):
    """
    测试集图数据结构
    功能：与训练集类似，用于模型评估
    """
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
        test_adjacency_matrix = torch.from_numpy(test_adjacency_matrix)
        test_node_attr_matrix = torch.from_numpy(test_node_attr_matrix)
        test_label_matrix = torch.from_numpy(test_label_matrix)
        return test_adjacency_matrix, test_node_attr_matrix, test_label_matrix

class GAT_Set(Dataset):
    """
    预测用图数据结构
    功能：仅包含图结构，不包含标签，用于模型推理阶段
    """
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

# ==================== 图注意力网络模型模块 ====================
# 功能：定义GAT网络各层和完整模型架构

class nconv(nn.Module):
    """
    图卷积操作基类
    功能：实现基本的图卷积运算
    """
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class GraphAttentionLayer(nn.Module):
    """
    图注意力层
    功能：实现多头注意力机制，计算节点间的注意力权重
    """
    def __init__(self, in_features, out_features, relu, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.relu = relu
        self.concat = concat

        # 权重矩阵初始化
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 注意力系数矩阵初始化
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
    """
    单层多头GAT模型
    功能：集成多个注意力头，实现特征融合
    """
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
    """
    完整GAT模型
    功能：包含图注意力层、全连接层和不确定性估计模块
    """
    def __init__(self, p, logvar, relu):
        super(GAT, self).__init__()
        self.p = p
        self.logvar = logvar

        # 图注意力模块
        self.graph_modules = nn.Sequential(OrderedDict([
            ('GAT_layer_0', GATModel_1layer(n_feat=1, n_hid=2, n_class=0, n_heads=2, relu=relu)),
        ]))

        # 全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=23 * 2 * 2, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.69)
        )

        # 预测前处理层
        self.pre = nn.Sequential(
            nn.Linear(in_features=128, out_features=16),
            nn.GELU(),
        )

        # 方差估计层
        self.var = nn.Sequential()

        # 预测输出层
        self.predict = nn.Sequential(
            nn.Linear(in_features=16, out_features=1),
            nn.ReLU(),
        )

        # 方差输出层
        self.get_var = nn.Sequential(
            nn.Linear(in_features=16, out_features=1),
        )

    def forward(self, node_attr_matrix, adjacency_matrix):
        adjacency_matrix = adjacency_matrix.float()
        train_x = node_attr_matrix.float()
        x = train_x[:, :23, :]

        # 图注意力特征提取
        for (name, module) in self.graph_modules.named_children():
            if 'GAT_layer' in name:
                x = module(x, adj=adjacency_matrix)
            else:
                x = module(x)

        x = x.view(x.size()[0], -1)
        xc = self.fc1(x)
        # 获取预测均值
        x = self.pre(xc)
        y = self.predict(x)

        # 获取预测方差
        xv = self.var(x)
        if self.var:
            var = self.get_var(xv)
        else:
            var = torch.zeros(y.size())
        return y, var, xc

# ==================== 损失函数模块 ====================
# 功能：定义模型训练所需的损失函数 

class Statistical_loss(nn.Module):
    """
    统计损失函数
    功能：结合预测误差和方差的不确定性感知损失函数
    """
    def __init__(self):
        super(Statistical_loss, self).__init__()

    def forward(self, gt, pred_y, logvar):
        gt = gt.cuda()
        pred_y = pred_y.cuda()
        logvar = logvar.cuda()
        loss = torch.mean(0.5*(torch.exp((-1)*logvar)) * (gt - pred_y)**2 + 0.5*logvar)
        return loss

# ==================== 训练与评估模块 ====================
# 功能：模型训练、验证和预测函数

def train(model, train_data_loader, epochs, optimizer, criterion, scheduler, num_ens=1, beta_type=0.1):
    """
    模型训练函数
    功能：执行GAT模型训练循环，包含前向传播、损失计算和反向传播
    参数：
        model: 待训练的GAT模型
        train_data_loader: 训练数据加载器
        epochs: 训练轮数
        optimizer: 优化器
        criterion: 损失函数
        scheduler: 学习率调度器
        num_ens: 集成模型数量
        beta_type: KL散度权重
    """
    print(' ')
    print("*** Training started! ***")
    print(' ')
    
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
    """
    Monte Carlo Dropout预测函数
    功能：通过多次采样获取预测均值、数据不确定性和模型不确定性
    参数：
        network: 训练好的GAT模型
        data_loader: 数据加载器
        dropout: 是否启用dropout
        mc_times: Monte Carlo采样次数
    返回：标签、预测值、数据不确定性、模型不确定性、总不确定性
    """
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

        # 计算数据不确定性
        a_u = np.sqrt(np.exp(np.mean(a_u, axis=2)))
        # 计算预测均值
        pred_mean = np.mean(pred_v, axis=2)
        # 计算模型不确定性
        e_u = np.sqrt(np.var(pred_v, axis=2))
        # 计算总不确定性
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
