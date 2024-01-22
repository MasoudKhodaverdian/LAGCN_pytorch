import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight

def read_data():
    A = pd.read_csv('G:/dissertation/important/LAGCN/data/drug_dis.csv', header=None).values
    Sr = pd.read_csv('G:/dissertation/important/LAGCN/data/drug_sim.csv', header=None).values
    Sd = pd.read_csv('G:/dissertation/important/LAGCN/data/dis_sim.csv', header=None).values

    A_np = np.array(A)
    Sr_np = np.array(Sr)
    Sd_np = np.array(Sd)

    return A_np, Sr_np, Sd_np

def xavier(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    r1 = -1 * init_range
    r2 = init_range
    initial = torch.FloatTensor(a, b).uniform_(r1, r2)
    return initial

def loss_function(inp,target):
    class_weights = compute_class_weight('balanced',np.unique(inp),inp.numpy())
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
    loss = criterion(inp,target)
    return loss