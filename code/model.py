import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import xavier

class LAGCN(nn.Module):
    def __init__(self,G,n_dimension,m_dimension,k=64,L=3): # normalizied G
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


    def forward(self, x): # x is H_0 in paper
        H_1 = self.act(torch.matmul(torch.matmul(self.G,x),self.W_0))
        H_2 = self.act(torch.matmul(torch.matmul(self.G,H_1),self.W_1))
        H_3 = self.act(torch.matmul(torch.matmul(self.G,H_2),self.W_2))
        H = self.a_1 * H_1 + self.a_2 * H_2 + self.a_3 * H_3 # H is combined of H_R and H_D
        H_R,H_D = torch.split(H, [self.n_dimension,self.m_dimension ])
        A_p = nn.Sigmoid()(torch.matmul(torch.matmul(H_R,self.W_p),torch.transpose(H_D,0,1)))
        return A_p





