import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight

def read_data():
    A = pd.read_csv('data/drug_dis.csv', header=None).values
    Sr = pd.read_csv('data/drug_sim.csv', header=None).values
    Sd = pd.read_csv('data/dis_sim.csv', header=None).values

    A_np = np.array(A)
    Sr_np = np.array(Sr)
    Sd_np = np.array(Sd)
    return A_np, Sr_np, Sd_np

def normalize_similarity_matrices(A_np, Sr_np, Sd_np):
    # Calculate diagonal matrices
    Dr = np.diag(np.sum(Sr_np, axis=1))
    Dd = np.diag(np.sum(Sd_np, axis=1))

    # Normalize similarity matrices
    Sr_norm = np.linalg.pinv(Dr) @ Sr_np @ np.linalg.pinv(Dr)
    Sd_norm = np.linalg.pinv(Dd) @ Sd_np @ np.linalg.pinv(Dd)
    return Sr_norm, Sd_norm

def construct_HNet(A_np,Sr_norm,Sd_norm):
    mat1 = np.hstack((Sr_np, A_np))
    mat2 = np.hstack((A_np.T, Sd_np))
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