import numpy as np
import pandas as pd

def read_data():
    A = pd.read_csv('G:/dissertation/important/LAGCN/data/drug_dis.csv', header=None).values
    Sr = pd.read_csv('G:/dissertation/important/LAGCN/data/drug_sim.csv', header=None).values
    Sd = pd.read_csv('G:/dissertation/important/LAGCN/data/dis_sim.csv', header=None).values

    A_np = np.array(A)
    Sr_np = np.array(Sr)
    Sd_np = np.array(Sd)

    return A_np, Sr_np, Sd_np
