from utils import read_data,normalizeAdjacency,construct_HNet,construct_Net,split_train_test,loss_function
from utils import evaluate
from train import train
import torch
from model import LAGCN
import copy

drug_dis_matrix,drug_sim,dis_sim = read_data()
G = construct_HNet(drug_dis_matrix,drug_sim*6,dis_sim*6)
normalized_G = torch.tensor(normalizeAdjacency(G)).float()
adj_nonzero = torch.count_nonzero(normalized_G)
model = LAGCN(normalized_G,drug_sim.shape[0],dis_sim.shape[0],adj_nonzero,k=64)
train_ind,test_ind = split_train_test(drug_dis_matrix)
train_matrix = copy.deepcopy(drug_dis_matrix)
train_matrix[test_ind] = 0
H_0 = construct_Net(train_matrix)
train_matrix = torch.tensor(train_matrix).float()
train_H_0 = torch.tensor(H_0).float()
drug_dis_matrix = drug_dis_matrix.astype(int)

model = train(model,3000,train_H_0,train_matrix,drug_dis_matrix,train_H_0)
model.eval()
pred = model(train_H_0)
pred = pred.detach().numpy()
aupr,auc,accuracy,f1 = evaluate(drug_dis_matrix[train_matrix==0].flatten(),pred[train_matrix==0].flatten())
print('aupr: ',aupr,' auc: ',auc)
print('f1: ',f1,'accuracy: ',accuracy)


