# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from scipy import sparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
import geatpy as ea  # Import geatpy library for genetic algorithm module
import csv
from sklearn import preprocessing
import numpy as np
import time
import pandas as pd
import sys

# Load data
start1 = time.perf_counter()
# ==================== LOGGING MODULE ====================
# Function: Simultaneously write console output to a file
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

# ==================== DATA LOADING AND PREPROCESSING MODULE ====================
# Function: Read data from Excel file and perform preprocessing
df = pd.read_excel("AM_Data.xlsx")
feature_name = [column for column in df][1:]
sample_data = df.values[:, :]  # Multi-scale dataset
Data_feature = sample_data[:, 1:-1]

# Thermodynamic data
Data_feature_CoarseningRate = sample_data[:, 1:14]
Data_target_CoarseningRate_ORI = sample_data[:, 14:-7]
Datadiffusion = Data_target_CoarseningRate_ORI[:, 1]
Data_target_CoarseningRate = np.column_stack([Data_target_CoarseningRate_ORI[:, 0],
                                              np.log(Datadiffusion),
                                              Data_target_CoarseningRate_ORI[:, 2],
                                              Data_target_CoarseningRate_ORI[:, 3]])

# COMSOL data
Data_feature_COMSOL = sample_data[:, 1:14]
Data_target_COMSOL = sample_data[:, 18:-5]

# Volume energy data
Data_feature_Energy = sample_data[:, 1:-3]
Data_feature_Energy = np.delete(Data_feature_Energy, [13, 14, 15, 16], axis=1)
Data_target_Energy = sample_data[:, -3]

# Hardness data
Data_feature_Hardness = sample_data[:, 1:-2]
Data_target_Hardness = sample_data[:, -2]

# Defect prediction data
Data_feature_II = sample_data[:, 14:24]
Data_target = sample_data[:, -1]
print('Data_feature = ', Data_feature_COMSOL.tolist())
print('Data_target = ', Data_target.tolist())
start_time1 = time.time()

# ==================== DATA STANDARDIZATION MODULE ====================
# Function: Standardize/normalize features and labels
ss_feature = preprocessing.StandardScaler()
Data_feature_nor = ss_feature.fit_transform(Data_feature)

# Thermodynamic data standardization
ss_feature_CoarseningRate = preprocessing.StandardScaler()
Data_feature_CoarseningRate = ss_feature_CoarseningRate.fit_transform(Data_feature_CoarseningRate)
ss_target_CoarseningRate = preprocessing.MinMaxScaler()
Data_target_CoarseningRate = ss_target_CoarseningRate.fit_transform(Data_target_CoarseningRate.reshape(-1, 4))

# COMSOL data standardization
ss_feature_COMSOL = preprocessing.StandardScaler()
Data_feature_COMSOL = ss_feature_COMSOL.fit_transform(Data_feature_COMSOL)
ss_target_COMSOL = preprocessing.MinMaxScaler()
Data_target_COMSOL = ss_target_COMSOL.fit_transform(Data_target_COMSOL.reshape(-1, 2))

# Volume energy data standardization
ss_feature_Energy = preprocessing.StandardScaler()
Data_feature_Energy = ss_feature_Energy.fit_transform(Data_feature_Energy)
ss_target_Energy = preprocessing.StandardScaler()
Data_target_Energy = ss_target_Energy.fit_transform(Data_target_Energy.reshape(-1, 1))

# Hardness data standardization
ss_feature_Hardness = preprocessing.StandardScaler()
Data_feature_Hardness = ss_feature_Hardness.fit_transform(Data_feature_Hardness)
ss_target_Hardness = preprocessing.StandardScaler()
Data_target_Hardness = ss_target_Hardness.fit_transform(Data_target_Hardness.reshape(-1, 1))

# Defect data standardization
ss_feature_II = preprocessing.StandardScaler()
Data_feature_II = ss_feature_II.fit_transform(Data_feature_II)

# ==================== GRAPH DATA PROCESSING MODULE ====================
# Function: Construct graph adjacency matrix and feature matrix

# Read adjacency matrix
neighbor_matrix = []
csv_file1 = csv.reader(open('data_matrix.csv', encoding='utf-8-sig'))
for content in csv_file1:
    content = list(map(float, content))
    if len(content) != 0:
        neighbor_matrix.append(content)

print('neighbor_matrix=', neighbor_matrix)
neighbor_matrix = np.array(neighbor_matrix)

# Function: Convert feature matrix to sparse matrix format
def manipulate_feature(feature, max_node, features):
    feature = feature.reshape(-1, 1)  # Convert to 1 column (32, 1)
    feature[:, [0]] = (feature[:, [0]])  #
    # Match maximum dimension of features
    result = np.zeros((max_node, features))  # Create zero matrix (32, 1)
    result[:feature.shape[0], :feature.shape[1]] = feature
    feature = result
    # Convert feature matrix to sparse matrix format: coordinates + values
    feature = sparse.csr_matrix(feature)
    return feature

# Function: Symmetrically normalize adjacency matrix and convert to sparse matrix
def normalize_adj(neighbor, max_node, feature):
    np.fill_diagonal(neighbor, 1)
    neighbor = sparse.csr_matrix(neighbor)
    return neighbor

# Function: Convert tensor to Variable
def tensor_to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x.float())


# Function: Convert Variable to numpy array
def variable_to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    return x


# Function: CNN feature processing
def cnn_feature(feature):
    feature = feature.reshape(-1, 1)  # Convert to 1 column (32, 1)
    feature = sparse.csr_matrix(feature)
    return feature

# ==================== GRAPH DATASET CLASS ====================
# Function: Build graph dataset for GAT prediction
class GAT_Set(Dataset):
    def __init__(self, train_x):
        max_node = 23
        num_features = 1
        neighbor_matrix_nor = neighbor_matrix

        # Generate graph structure from split data
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
        self.train_node_attr_matrix = np.array(train_node_attr_matrix)  # Node numbers + feature values

    def __len__(self):
        return len(self.train_adjacency_matrix)

    def __getitem__(self, idx):
        train_adjacency_matrix = self.train_adjacency_matrix[idx].todense()
        train_node_attr_matrix = self.train_node_attr_matrix[idx].todense()
        train_adjacency_matrix = torch.from_numpy(train_adjacency_matrix)
        train_node_attr_matrix = torch.from_numpy(train_node_attr_matrix)
        return train_adjacency_matrix, train_node_attr_matrix

# Function: GAT prediction function supporting MC Dropout uncertainty estimation
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

        a_u = np.sqrt(np.exp(np.mean(a_u, axis=2)))  # Epistemic uncertainty
        pred_mean = np.mean(pred_v, axis=2)  # Prediction mean
        e_u = np.sqrt(np.var(pred_v, axis=2))  # Aleatoric uncertainty
        un = a_u + e_u  # Total uncertainty

        pre_list.extend(pred_mean)
        al_list.extend(a_u)
        ep_list.extend(e_u)
        un_list.extend(un)

    pre_list = np.array(pre_list)
    al_list = np.array(al_list)
    ep_list = np.array(ep_list)
    un_list = np.array(un_list)
    return pre_list, ep_list, al_list, un_list


# ==================== THERMODYNAMIC DATA PREDICTION MODEL MODULE ====================
# Function: Thermodynamic dataset
class CoarseningRate_Set(Dataset):
    def __init__(self, train_x):
        for i in range(len(train_x)):
            train_feature = cnn_feature(train_x[i])
            train_multiple_feature = [train_feature]

            if i == 0:
                train_node_attr_matrix = train_multiple_feature
            else:
                train_node_attr_matrix = np.concatenate((train_node_attr_matrix, train_multiple_feature))
        self.train_node_attr_matrix = np.array(train_node_attr_matrix)  # Node numbers + feature values

    def __len__(self):
        return len(self.train_node_attr_matrix)

    def __getitem__(self, idx):
        train_node_attr_matrix = self.train_node_attr_matrix[idx].todense()
        train_node_attr_matrix = torch.from_numpy(train_node_attr_matrix)
        return train_node_attr_matrix


# ==================== COMSOL PREDICTION MODEL MODULE ====================
# Function: COMSOL dataset
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


# ==================== HARDNESS PREDICTION MODEL MODULE ====================
# Function: Hardness dataset
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


# ==================== ENERGY PREDICTION MODEL MODULE ====================
# Function: Energy dataset
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

# Function: Prediction function
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

# ==================== MODEL LOADING MODULE ====================
# Function: Load pre-trained model weights
## Load UQ-KGAT model module and intermediate machine learning models
torch.manual_seed(3416)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3416)
    torch.cuda.manual_seed_all(3416)
torch.backends.cudnn.deterministic = True

# Load each pre-trained model
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

# ==================== INTEGRATED PREDICTION FUNCTION ====================
# Function: Integrate all models for end-to-end prediction
def PreFunc(feature):  # Objective function
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

# ==================== MULTI-OBJECTIVE OPTIMIZATION MODULE ====================
# Function: Define NSGA-II optimization problem
class MyProblem(ea.Problem):  # Inherit from parent class Problem
    def __init__(self):
        name = 'MyProblem'  # Initialize name (function name, can be set arbitrarily)
        M = 2  # Initialize M (number of objectives)
        maxormins = [1, 1]  # Initialize maxormins (list of minimization/maximization flags: 1=minimize, -1=maximize)
        Dim = 13  # Initialize Dim (number of decision variables)
        varTypes = [0] * Dim  # Initialize varTypes (type of decision variables: 0=continuous, 1=discrete)

        lbin = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Initial lower bounds
        ubin = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Initial upper bounds

        # Set value ranges for 13 design variables
        lb = [3.49, 8, 5.05, 0.03, 0.66, 0.01, 2.25, 0.06, 0.01, 0, 0, 4, 600]  # Initial lower bounds
        ub = [6.44, 14.6, 15.93, 2, 5, 3.28, 8, 0.21, 0.06, 2.1, 0.03, 14, 1600]  # Initial upper bounds

        # Call parent class constructor to complete instantiation
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # Objective function
        x = pop.Phen
        pop.ObjV = np.zeros((pop.Phen.shape[0], self.M))
        # Call integrated prediction function
        All_Info = PreFunc(x)
        GA_dAta.append(All_Info)

        pop.Phen = All_Info
        # Objective 1: Minimize defect prediction value
        pop.ObjV[:, 0] = All_Info[:, -4].flatten()
        # Objective 2: Minimize total uncertainty
        pop.ObjV[:, 1] = All_Info[:, -1].flatten()


# ==================== MAIN PROGRAM ====================
# Function: Run multi-objective optimization algorithm
if __name__ == "__main__":
    for kk in range(20):  # Run 20 times
        GA_dAta = []
        combined_Phen = None
        combined_ObjV = None
        num_runs = 1  # Set number of runs

        for run in range(num_runs):
            problem = MyProblem()  # Create problem object
            """==================================POPULATION SETTINGS================================"""
            Encoding = 'RI'  # Encoding method
            NIND = 50  # Population size
            Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # Create region descriptor
            population = ea.Population(Encoding, Field, NIND)  # Instantiate population object
            """=================================ALGORITHM PARAMETER SETTINGS=============================="""
            myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # Instantiate NSGA-II algorithm template
            myAlgorithm.mutOper.Pm = 0.2  # Modify mutation probability
            myAlgorithm.recOper.XOVR = 0.9  # Modify crossover probability
            myAlgorithm.MAXGEN = 1500  # Maximum number of generations
            """============================CALL ALGORITHM TEMPLATE FOR POPULATION EVOLUTION========================="""
            [NDSet, population] = myAlgorithm.run()  # Execute algorithm template to get Pareto optimal set NDSet

            if combined_Phen is None:
                combined_Phen = NDSet.Phen
                combined_ObjV = NDSet.ObjV
            else:
                combined_Phen = np.vstack((combined_Phen, NDSet.Phen))
                combined_ObjV = np.vstack((combined_ObjV, NDSet.ObjV))

        combined_NDSet = ea.Population(Encoding, Field, 0)  # Create empty population object
        combined_NDSet.Phen = combined_Phen
        combined_NDSet.ObjV = combined_ObjV

        # Save optimization results
        GA_dAta = np.array(GA_dAta)
        GA_dAta = pd.DataFrame(GA_dAta.reshape(75000, 27))
        GA_dAta.columns = feature_name + ['epistemic', 'aleatoric', 'total']
        GA_dAta.to_csv('Results/GA_dAta/GA_dAta_%i.csv' % kk)

        Best_x = combined_NDSet.Phen
        Best_y1 = combined_NDSet.ObjV[:, -2].reshape(-1, 1)
        Best_y2 = combined_NDSet.ObjV[:, -1].reshape(-1, 1)

        # Save Pareto optimal solutions
        feature_name_all = feature_name + ['ep_pre', 'al_pre', 'un_pre']
        data2 = pd.DataFrame(Best_x)
        data2.columns = feature_name_all
        data2.to_csv('Results/all targets %i.csv' % kk)

        # Output statistical information
        print('Time used: %s seconds' % (myAlgorithm.passTime))
        print('Number of non-dominated individuals: %s' % (NDSet.sizes))
        print('Number of Pareto front points found per unit time: %s' % (int(NDSet.sizes // myAlgorithm.passTime)))
