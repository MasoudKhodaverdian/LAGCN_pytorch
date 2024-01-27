import torch
from utils import loss_function,loss_function_2
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,CyclicLR
import copy
import pickle
from tqdm import tqdm
from functools import partial

tqdm = partial(tqdm, position=0, leave=True)

def train(model,n_epochs,inp,train_matrix):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    inp = inp.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.008)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')
    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01,step_size_up=int(n_epochs/2),mode='exp_range',gamma=0.8,cycle_momentum=False)
    min_loss = float('inf')
    for epoch in tqdm(range(n_epochs),desc="traning"):
        optimizer.zero_grad()
        target = model(inp)
        loss = loss_function_2(train_matrix,target)
        # scheduler.step(loss)

        loss.backward()
        if loss.item() < min_loss :
            min_loss = loss.item()
            best_model = copy.deepcopy(model)
        optimizer.step()
        scheduler.step()
        if epoch % 100 == 1 :
            print('epoch: ',epoch,' loss: ',loss.item())
    print('minimum loss: ',min_loss)

    # try:
    #     pre_best = pickle.load('best_loss.pkl')
    # except:
    #     pre_best = float('inf')
    # if min_loss < pre_best :
    #     pickle.dump('best_loss.pkl', min_loss )
    #     torch.save(best_model, 'best_model.pth')

    return best_model


