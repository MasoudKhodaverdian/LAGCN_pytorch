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


# import torch
# from torch.optim.lr_scheduler import CyclicLR
#
# # neural network def
# model = YourModel()
#
# #opotimaizer and lr
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
# # cycling lr definition
# scheduler = CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=2000, mode='triangular')
#
# # neural network training with clr
# for epoch in range(num_epochs):
#     for batch_idx, (data, targets) in enumerate(train_loader):
#         # loss and output
#         outputs = model(data)
#         loss = criterion(outputs, targets)
#
#         # removing pre grad
#         optimizer.zero_grad()
#
#         # loss computation
#         loss.backward()
#
#         # weights updating
#         optimizer.step()
#         scheduler.step()