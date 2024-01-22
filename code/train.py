import torch
from utils import loss_function
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import pickle
from tqdm import tqdm

def train(model,n_epochs,inp):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    inp = inp.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    min_loss = float('inf')
    for epoch in tqdm(range(n_epochs),desc="traning"):
        target = model(inp)
        loss = loss_function(inp,target)
        if loss < min_loss :
            loss = min_loss
            best_model = copy.deepcopy(model)
        scheduler.step(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    try:
        pre_best = pickle.load('best_loss.pkl')
    except:
        pre_best = float('inf')
    if min_loss < pre_best :
        pickle.dump('best_loss.pkl', min_loss )
        torch.save(best_model, 'best_model.pth')

    return best_model

