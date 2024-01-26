import torch
from utils import loss_function,loss_function_2
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import pickle
from tqdm import tqdm

def train(model,n_epochs,inp,train_matrix):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    inp = inp.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    min_loss = float('inf')
    for epoch in tqdm(range(n_epochs),desc="traning"):
        optimizer.zero_grad()
        target = model(inp)
        loss = loss_function_2(train_matrix,target)
        # print(loss)
        scheduler.step(loss)
        loss.backward()
        if loss.item() < min_loss :
            min_loss = loss.item()
            best_model = copy.deepcopy(model)
        optimizer.step()
    print(loss.item())

    # try:
    #     pre_best = pickle.load('best_loss.pkl')
    # except:
    #     pre_best = float('inf')
    # if min_loss < pre_best :
    #     pickle.dump('best_loss.pkl', min_loss )
    #     torch.save(best_model, 'best_model.pth')

    return best_model

