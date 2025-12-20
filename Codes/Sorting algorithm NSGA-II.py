# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from scipy import sparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
import geatpy as ea  # 导入geatpy库 调用遗传算法模块
import csv
from sklearn import preprocessing
import numpy as np
import time
import pandas as pd
import sys

# 导入数据
start1 = time.perf_counter()


# ==================== 日志模块 ====================
# 功能：将控制台输出同时写入文件
class Logger(object):
    def __init__(self, file="Default.log"):
        self.terminal = sys.stdout
        self.log = open(file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()


sys.stdout = Logger('Results/Console_log.txt')

# ==================== 数据加载与预处理模块 ====================
# 功能：从Excel文件读取数据并进行预处理
df = pd.read_excel("../../AM_Data_0906.xlsx")
feature_name = [column for column in df][1:]
sample_data = df.values[:, :]  # 多尺度数据集
Data_feature = sample_data[:, 1:-1]

# Thermodynamic数据
Data_feature_CoarseningRate = sample_data[:, 1:14]
Data_target_CoarseningRate_ORI = sample_data[:, 14:-7]
Datadiffusion = Data_target_CoarseningRate_ORI[:, 1]
Data_target_CoarseningRate = np.column_stack([Data_target_CoarseningRate_ORI[:, 0],
                                              np.log(Datadiffusion),
                                              Data_target_CoarseningRate_ORI[:, 2],
                                              Data_target_CoarseningRate_ORI[:, 3]])

# COMSOL数据
Data_feature_COMSOL = sample_data[:, 1:14]
Data_target_COMSOL = sample_data[:, 18:-5]

# 体能量数据
Data_feature_Energy = sample_data[:, 1:-3]
Data_feature_Energy = np.delete(Data_feature_Energy, [13, 14, 15, 16], axis=1)
Data_target_Energy = sample_data[:, -3]

# 硬度数据
Data_feature_Hardness = sample_data[:, 1:-2]
Data_target_Hardness = sample_data[:, -2]

# 缺陷预测数据
Data_feature_II = sample_data[:, 14:24]
Data_target = sample_data[:, -1]
print('Data_feature = ', Data_feature_COMSOL.tolist())
print('Data_target = ', Data_target.tolist())
start_time1 = time.time()

# ==================== 数据标准化模块 ====================
# 功能：对特征和标签进行标准化/归一化处理
ss_feature = preprocessing.StandardScaler()
Data_feature_nor = ss_feature.fit_transform(Data_feature)

# Thermodynamic数据标准化
ss_feature_CoarseningRate = preprocessing.StandardScaler()
Data_feature_CoarseningRate = ss_feature_CoarseningRate.fit_transform(Data_feature_CoarseningRate)
ss_target_CoarseningRate = preprocessing.MinMaxScaler()
Data_target_CoarseningRate = ss_target_CoarseningRate.fit_transform(Data_target_CoarseningRate.reshape(-1, 4))

# COMSOL数据标准化
ss_feature_COMSOL = preprocessing.StandardScaler()
Data_feature_COMSOL = ss_feature_COMSOL.fit_transform(Data_feature_COMSOL)
ss_target_COMSOL = preprocessing.MinMaxScaler()
Data_target_COMSOL = ss_target_COMSOL.fit_transform(Data_target_COMSOL.reshape(-1, 2))

# 体能量数据标准化
ss_feature_Energy = preprocessing.StandardScaler()
Data_feature_Energy = ss_feature_Energy.fit_transform(Data_feature_Energy)
ss_target_Energy = preprocessing.StandardScaler()
Data_target_Energy = ss_target_Energy.fit_transform(Data_target_Energy.reshape(-1, 1))

# 硬度数据标准化
ss_feature_Hardness = preprocessing.StandardScaler()
Data_feature_Hardness = ss_feature_Hardness.fit_transform(Data_feature_Hardness)
ss_target_Hardness = preprocessing.StandardScaler()
Data_target_Hardness = ss_target_Hardness.fit_transform(Data_target_Hardness.reshape(-1, 1))

# 缺陷数据标准化
ss_feature_II = preprocessing.StandardScaler()
Data_feature_II = ss_feature_II.fit_transform(Data_feature_II)

# ==================== 图数据处理模块 ====================
# 功能：构建图的邻接矩阵和特征矩阵

# 读取邻接矩阵
neighbor_matrix = []
csv_file1 = csv.reader(open('../../data_matrix.csv', encoding='utf-8-sig'))
for content in csv_file1:
    content = list(map(float, content))
    if len(content) != 0:
        neighbor_matrix.append(content)

print('neighbor_matrix=', neighbor_matrix)
neighbor_matrix = np.array(neighbor_matrix)

# 功能：将特征矩阵转换为稀疏矩阵格式
def manipulate_feature(feature, max_node, features):
    feature = feature.reshape(-1, 1)  # 转换成1列 (32, 1)
    feature[:, [0]] = (feature[:, [0]])  #
    # 匹配特征features的最大维度
    result = np.zeros((max_node, features))  # 创建(32, 1)的零矩阵
    result[:feature.shape[0], :feature.shape[1]] = feature
    feature = result
    # 将特征矩阵转换成稀疏矩阵 形式: 坐标 + 值
    feature = sparse.csr_matrix(feature)
    return feature

# 功能：对邻接矩阵进行对称归一化并转换为稀疏矩阵
def normalize_adj(neighbor, max_node, feature):
    np.fill_diagonal(neighbor, 1)
    neighbor = sparse.csr_matrix(neighbor)
    return neighbor

# 功能：张量转换为Variable
def tensor_to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x.float())


# 功能：Variable转换为numpy数组
def variable_to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    return x


# 功能：CNN特征处理
def cnn_feature(feature):
    feature = feature.reshape(-1, 1)  # 转换成1列 (32, 1)
    feature = sparse.csr_matrix(feature)
    return feature

# ==================== 图数据集类 ====================
# 功能：构建GAT预测用图数据集
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
                    np.concatenate((train_adjacency_matrix, train_multiple_neighbor)), np.concatenate(
                        (train_node_attr_matrix, train_multiple_feature))
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

# 功能：GAT预测函数，支持MC Dropout不确定性估计
def GAT_PRE(network, data_loader, dropout=True, mc_times=64):
    pre_list, ep_list, al_list, un_list = [], [], [], []

    for batch_id, (adjacency_matrix, node_attr_matrix) in enumerate(data_loader):
        adjacency_matrix = tensor_to_variable(adjacency_matrix)
        inputs = tensor_to_variable(node_attr_matrix)

        pred_v = np.zeros((len(inputs), 1, mc_times))
        a_u = np.zeros((len(inputs), 1, mc_times))
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

        a_u = np.sqrt(np.exp(np.mean(a_u, axis=2)))  # 认知不确定性
        pred_mean = np.mean(pred_v, axis=2)  # 预测均值
        e_u = np.sqrt(np.var(pred_v, axis=2))  # 偶然不确定性
        un = a_u + e_u  # 总不确定性

        pre_list.extend(pred_mean)
        al_list.extend(a_u)
        ep_list.extend(e_u)
        un_list.extend(un)

    pre_list = np.array(pre_list)
    al_list = np.array(al_list)
    ep_list = np.array(ep_list)
    un_list = np.array(un_list)
    return pre_list, ep_list, al_list, un_list


# ==================== Thermodynamic 数据预测模型模块 ====================
# 功能：Thermodynamic 数据集
class CoarseningRate_Set(Dataset):
    def __init__(self, train_x):
        for i in range(len(train_x)):
            train_feature = cnn_feature(train_x[i])
            train_multiple_feature = [train_feature]

            if i == 0:
                train_node_attr_matrix = train_multiple_feature
            else:
                train_node_attr_matrix = np.concatenate((train_node_attr_matrix, train_multiple_feature))
        self.train_node_attr_matrix = np.array(train_node_attr_matrix)  # 节点编号+特征值

    def __len__(self):
        return len(self.train_node_attr_matrix)

    def __getitem__(self, idx):
        train_node_attr_matrix = self.train_node_attr_matrix[idx].todense()
        train_node_attr_matrix = torch.from_numpy(train_node_attr_matrix)
        return train_node_attr_matrix


# ==================== COMSOL预测模型模块 ====================
# 功能：COMSOL数据集
class COMSOL_Set(Dataset):
    def __init__(self, train_x):
        for i in range(len(train_x)):
            train_feature = cnn_feature(train_x[i])
            train_multiple_feature = [train_feature]

            if i == 0:
                train_node_attr_matrix = train_multiple_feature
            else:
                train_node_attr_matrix = np.concatenate((train_node_attr_matrix, train_multiple_feature))
        self.train_node_attr_matrix = np.array(train_node_attr_matrix)

    def __len__(self):
        return len(self.train_node_attr_matrix)

    def __getitem__(self, idx):
        train_node_attr_matrix = self.train_node_attr_matrix[idx].todense()
        train_node_attr_matrix = torch.from_numpy(train_node_attr_matrix)
        return train_node_attr_matrix


# ==================== 硬度预测模型模块 ====================
# 功能：硬度数据集
class Hardness_Set(Dataset):
    def __init__(self, train_x):
        for i in range(len(train_x)):
            train_feature = cnn_feature(train_x[i])
            train_multiple_feature = [train_feature]

            if i == 0:
                train_node_attr_matrix = train_multiple_feature
            else:
                train_node_attr_matrix = np.concatenate((train_node_attr_matrix, train_multiple_feature))
        self.train_node_attr_matrix = np.array(train_node_attr_matrix)

    def __len__(self):
        return len(self.train_node_attr_matrix)

    def __getitem__(self, idx):
        train_node_attr_matrix = self.train_node_attr_matrix[idx].todense()
        train_node_attr_matrix = torch.from_numpy(train_node_attr_matrix)
        return train_node_attr_matrix


# ==================== 能量预测模型模块 ====================
# 功能：能量数据集
class Energy_Set(Dataset):
    def __init__(self, train_x):
        for i in range(len(train_x)):
            train_feature = cnn_feature(train_x[i])
            train_multiple_feature = [train_feature]

            if i == 0:
                train_node_attr_matrix = train_multiple_feature
            else:
                train_node_attr_matrix = np.concatenate((train_node_attr_matrix, train_multiple_feature))
        self.train_node_attr_matrix = np.array(train_node_attr_matrix)

    def __len__(self):
        return len(self.train_node_attr_matrix)

    def __getitem__(self, idx):
        train_node_attr_matrix = self.train_node_attr_matrix[idx].todense()
        train_node_attr_matrix = torch.from_numpy(train_node_attr_matrix)
        return train_node_attr_matrix

# 功能：预测函数
def Indicator_PRE(model, data_loader):
    model.eval()
    if data_loader is None:
        return None, None

    y_pre_list = []
    for batch_id, (node_attr_matrix) in enumerate(data_loader):
        node_attr_matrix = tensor_to_variable(node_attr_matrix)
        y_pre = model(node_attr_matrix=node_attr_matrix)
        y_pre_list.extend(variable_to_numpy(y_pre))
    y_pre_list = np.array(y_pre_list)
    return y_pre_list

# ==================== 模型加载模块 ====================
# 功能：加载预训练好的模型权重
## 加载UQ-KGAT模型模块以及中间机器学习模型
torch.manual_seed(3416)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3416)
    torch.cuda.manual_seed_all(3416)
torch.backends.cudnn.deterministic = True

# 加载各个预训练模型
bdgcn = GAT(p=0, logvar=0.3, relu=0.017).cuda()
bdgcn.load_state_dict(torch.load('../Results/model state/gat_state.pth'))

coarseningRate_PMcnn = CoarseningRate_CNN().cuda()
coarseningRate_PMcnn.load_state_dict(torch.load('CoarseningRate Model/Results/model state/CoseningRate_state.pth'))

comsol_cnn = COMSOL_CNN().cuda()
comsol_cnn.load_state_dict(torch.load('Comsol Model/Results/model state/Comsol_state.pth'))

energy_cnn = Energy_CNN().cuda()
energy_cnn.load_state_dict(torch.load('Volume Energy Model/Results/model state/Volume energy_state.pth'))

hardness_cnn = Hardness_CNN().cuda()
hardness_cnn.load_state_dict(torch.load('Hardness Model/Results/model state/Hardness_state.pth'))

# ==================== 集成预测函数 ====================
# 功能：整合所有模型进行端到端预测
def PreFunc(feature):  # 目标函数
    # COMSOL_Feature Predict
    x = ss_feature_COMSOL.transform(feature)
    x_2 = np.zeros((feature.shape[0], 5))

    feature_I = np.column_stack([x, x_2])
    Comsol_data = COMSOL_Set(feature_I)
    Comsol_dataloader = DataLoader(Comsol_data, batch_size=64)
    COMSOL_II = Indicator_PRE(comsol_cnn, Comsol_dataloader)
    Comsol_feature = ss_target_COMSOL.inverse_transform(COMSOL_II)
    COMSOL_C = Comsol_feature[:, 0] * Comsol_feature[:, 1]
    COMSOL_Ra = Comsol_feature[:, 1] / Comsol_feature[:, 0]
    Comsol_feature_ALL = np.column_stack([Comsol_feature, COMSOL_C.reshape(-1, 1), COMSOL_Ra.reshape(-1, 1)])

    # Thermodynamic_PM Predict
    CoarseningRate_data = CoarseningRate_Set(feature_I)
    CoarseningRate_dataloader = DataLoader(CoarseningRate_data, batch_size=64)
    CoarseningRate_PM = Indicator_PRE(coarseningRate_PMcnn, CoarseningRate_dataloader)
    CoarseningRate_feature_ALL = ss_target_CoarseningRate.inverse_transform(CoarseningRate_PM)
    CoarseningRate_feature_ALL[:, 1] = np.exp(CoarseningRate_feature_ALL[:, 1])

    # Volume_Energy Predict
    feature_energy_all = np.column_stack([feature, Comsol_feature_ALL])
    feature_energy_all = ss_feature_Energy.transform(feature_energy_all)
    x_zero_energy = np.zeros((feature.shape[0], 8))
    feature_energy_all = np.column_stack([feature_energy_all, x_zero_energy])
    Energy_data = Energy_Set(feature_energy_all)
    Energy_dataloader = DataLoader(Energy_data, batch_size=64)
    Energy = Indicator_PRE(energy_cnn, Energy_dataloader)
    Energy = ss_target_Energy.inverse_transform(Energy)

    # Hardness Predict
    feature_hardness_all = np.column_stack([feature, CoarseningRate_feature_ALL, Comsol_feature_ALL, Energy])
    feature_hardness_all = ss_feature_Hardness.transform(feature_hardness_all)
    x_zero_hardness = np.zeros((feature.shape[0], 7))
    feature_hardness_all = np.column_stack([feature_hardness_all, x_zero_hardness])
    Hardenss_data = Hardness_Set(feature_hardness_all)
    Hardenss_dataloader = DataLoader(Hardenss_data, batch_size=64)
    Hardness = Indicator_PRE(hardness_cnn, Hardenss_dataloader)
    Hardness = ss_target_Hardness.inverse_transform(Hardness)

    # Defects Predict
    feature_all = np.column_stack([feature, CoarseningRate_feature_ALL, Comsol_feature_ALL, Energy, Hardness])
    feature_all = ss_feature.transform(feature_all)
    BDGCN_data = GAT_Set(feature_all)
    BDGCN_dataloader = DataLoader(BDGCN_data, batch_size=64)
    y_pre, ep_pre, al_pre, un_pre = GAT_PRE(bdgcn, BDGCN_dataloader)
    All_Info = np.column_stack(
        [feature, CoarseningRate_feature_ALL, Comsol_feature_ALL, Energy, Hardness, y_pre, ep_pre, al_pre, un_pre])
    return All_Info

# ==================== 多目标优化模块 ====================
# 功能：定义NSGA-II优化问题
class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 13  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）

        lbin = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 初始下界
        ubin = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 初始上界

        # 设置13个设计变量的取值范围
        lb = [3.49, 8, 5.05, 0.03, 0.66, 0.01, 2.25, 0.06, 0.01, 0, 0, 4, 600]  # 初始下界
        ub = [6.44, 14.6, 15.93, 2, 5, 3.28, 8, 0.21, 0.06, 2.1, 0.03, 14, 1600]  # 初始上界

        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        x = pop.Phen
        pop.ObjV = np.zeros((pop.Phen.shape[0], self.M))
        # 调用集成预测函数
        All_Info = PreFunc(x)
        GA_dAta.append(All_Info)

        pop.Phen = All_Info
        # 目标1：缺陷预测值最小化
        pop.ObjV[:, 0] = All_Info[:, -4].flatten()
        # 目标2：总不确定性最小化
        pop.ObjV[:, 1] = All_Info[:, -1].flatten()


# ==================== 主程序 ====================
# 功能：运行多目标优化算法
if __name__ == "__main__":
    for kk in range(20):  # 运行20次
        GA_dAta = []
        combined_Phen = None
        combined_ObjV = None
        num_runs = 1  # 设置多次运行的次数

        for run in range(num_runs):
            problem = MyProblem()  # 生成问题对象
            """==================================种群设置================================"""
            Encoding = 'RI'  # 编码方式
            NIND = 50  # 种群规模
            Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
            population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象
            """=================================算法参数设置=============================="""
            myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化NSGA-II算法模板对象
            myAlgorithm.mutOper.Pm = 0.2  # 修改变异算子的变异概率
            myAlgorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
            myAlgorithm.MAXGEN = 1500  # 最大进化代数
            """============================调用算法模板进行种群进化========================="""
            [NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到帕累托最优解集NDSet

            if combined_Phen is None:
                combined_Phen = NDSet.Phen
                combined_ObjV = NDSet.ObjV
            else:
                combined_Phen = np.vstack((combined_Phen, NDSet.Phen))
                combined_ObjV = np.vstack((combined_ObjV, NDSet.ObjV))

        combined_NDSet = ea.Population(Encoding, Field, 0)  # 创建一个空的种群对象
        combined_NDSet.Phen = combined_Phen
        combined_NDSet.ObjV = combined_ObjV

        # 保存优化结果
        GA_dAta = np.array(GA_dAta)
        GA_dAta = pd.DataFrame(GA_dAta.reshape(75000, 27))
        GA_dAta.columns = feature_name + ['epistemic', 'aleatoric', 'total']
        GA_dAta.to_csv('Results/GA_dAta/GA_dAta_%i.csv' % kk)

        Best_x = combined_NDSet.Phen
        Best_y1 = combined_NDSet.ObjV[:, -2].reshape(-1, 1)
        Best_y2 = combined_NDSet.ObjV[:, -1].reshape(-1, 1)

        # 保存帕累托最优解
        feature_name_all = feature_name + ['ep_pre', 'al_pre', 'un_pre']
        data2 = pd.DataFrame(Best_x)
        data2.columns = feature_name_all
        data2.to_csv('Results/all targets %i.csv' % kk)

        # 输出统计信息
        print('用时：%s 秒' % (myAlgorithm.passTime))
        print('非支配个体数：%s 个' % (NDSet.sizes))
        print('单位时间找到帕累托前沿点个数：%s 个' % (int(NDSet.sizes // myAlgorithm.passTime)))