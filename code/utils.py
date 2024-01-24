import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from scipy.linalg import fractional_matrix_power
import random

def read_data():
    # A = pd.read_csv('../data/drug_dis.csv', header=None).values
    # Sr = pd.read_csv('../data/drug_sim.csv', header=None).values
    # Sd = pd.read_csv('../data/dis_sim.csv', header=None).values

    A_np = np.loadtxt('../data/drug_dis.csv', delimiter=',')
    Sr_np = np.loadtxt('../data/drug_sim.csv', delimiter=',')
    Sd_np = np.loadtxt('../data/dis_sim.csv', delimiter=',')
    return A_np, Sr_np, Sd_np

def normalizeAdjacency(W): # input matrix must be a symmetric matrix

    assert W.shape[0] == W.shape[1]
    d = np.sum(W, axis = 1)
    d = 1/np.sqrt(d)
    D = np.diag(d)
    return D @ W @ D

def normalize_similarity_matrices(A_np, Sr_np, Sd_np):
    # Calculate diagonal matrices
    Dr = np.diag(np.sum(Sr_np, axis=1))
    Dd = np.diag(np.sum(Sd_np, axis=1))

    # Normalize similarity matrices
    # Sr_norm = np.linalg.pinv(Dr) @ Sr_np @ np.linalg.pinv(Dr)
    # Sd_norm = np.linalg.pinv(Dd) @ Sd_np @ np.linalg.pinv(Dd)
    Sr_norm = fractional_matrix_power(Dr, -0.5) @ Sr_np @ fractional_matrix_power(Dr, -0.5)
    Sd_norm = fractional_matrix_power(Dd, -0.5) @ Sr_np @ fractional_matrix_power(Dd, -0.5)
    return Sr_norm, Sd_norm

def construct_HNet(A_np,Sr_norm,Sd_norm):
    mat1 = np.hstack((Sr_norm, A_np))
    mat2 = np.hstack((A_np.T, Sd_norm))
    return np.vstack((mat1, mat2))

def construct_Net(A_np):
    drug_matrix = np.matrix(
        np.zeros((A_np.shape[0], A_np.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((A_np.shape[1], A_np.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, A_np))
    mat2 = np.hstack((A_np.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj

def xavier(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    r1 = -1 * init_range
    r2 = init_range
    initial = torch.FloatTensor(input_dim, output_dim).uniform_(r1, r2)
    return initial


def loss_function(inp,target):
    class_weights = compute_class_weight('balanced',np.unique(inp),inp.numpy())
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
    loss = criterion(inp,target)
    return loss

def split_train_test(drug_dis_matrix,ratio=0.8):
    index_matrix = np.mat(np.where(drug_dis_matrix == 1))
    # print(index_matrix)
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.shuffle(random_index)
    ind = int(association_nam * ratio)
    train = tuple(np.array(random_index[:ind]).T)
    test = tuple(np.array(random_index[ind:]).T)
    return train,test # these are index we can use drug_dis_matrix[test]

