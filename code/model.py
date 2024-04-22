import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import xavier,node_dropout

class LAGCN(nn.Module):
    def __init__(self,G,n_dimension,m_dimension,adj_nonzero,k=64,L=3,dropout=0.2): # normalizied G
        super(LAGCN, self).__init__()
        self.n_dimension = n_dimension
        self.m_dimension = m_dimension
        self.L = L
        self.G = G
        self.W_0 = torch.nn.Parameter(xavier(n_dimension+m_dimension,k))
        self.W_1 = torch.nn.Parameter(xavier(k,k))
        self.W_2 = torch.nn.Parameter(xavier(k,k))
        self.W_p = torch.nn.Parameter(xavier(k,k))
        self.a_1 = torch.nn.Parameter(torch.tensor(1/2))
        self.a_2 = torch.nn.Parameter(torch.tensor(1/3))
        self.a_3 = torch.nn.Parameter(torch.tensor(1/4))
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout)
        self.adj_nonzero = adj_nonzero


    def forward(self, x): # x is H_0 in paper
        # x = node_dropout(x,0.6,self.adj_nonzero)
        # x = nn.Dropout(0.2)(x)
        H_1 = self.act(torch.matmul(torch.matmul(self.G,x),self.W_0))
        # H_1 = self.drop(H_1)
        H_2 = self.act(torch.matmul(torch.matmul(self.G,H_1),self.W_1))
        # H_2 = self.drop(H_2)
        H_3 = self.act(torch.matmul(torch.matmul(self.G,H_2),self.W_2))
        # H_3 = self.drop(H_3)
        H = self.a_1 * H_1 + self.a_2 * H_2 + self.a_3 * H_3 # H is combined of H_R and H_D
        # H = self.drop(H)
        H_R,H_D = torch.split(H, [self.n_dimension,self.m_dimension ])
        A_p = nn.Sigmoid()(torch.matmul(torch.matmul(H_R,self.W_p),torch.transpose(H_D,0,1)))
        return A_p





