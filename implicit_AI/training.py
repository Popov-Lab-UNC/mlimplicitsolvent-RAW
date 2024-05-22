import torch
import torch.cuda
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from tqdm import tqdm
import numpy as np
from torchmdnet.models.tensornet import TensorNet

print(torch.cuda.is_available())

input = QM9("")
data2 = []
dataset = []

with open("solvation_final.pkl", 'rb') as file:
    data = pkl.load(file)

for force in tqdm(data):
    if hasattr(force, '_value'):  
        force = force._value
        if len(force) < 30:
            padding = np.zeros((30-len(force), force.shape[1]))
        force = torch.tensor(force)
    data2.append(force)
        

for i in tqdm(input):
    idx = int(i.name.split('_')[-1]) - 1
    if not isinstance(data2[idx], type(-1)):
        if(len(data[idx]) == len(i.z)):
            i.y = data2[idx]
            dataset.append(i)
    

test_dataset = dataset[:10000]
val_dataset = dataset[10000:20000]
train_dataset = dataset[20000:]
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


model = TensorNet(hidden_channels=128, 
                  num_layers = 3, 
                  trainable_rbf = True, 
                  activation = 'tanh', 
                  cutoff_upper = 10.0, 
                  equivariance_invariance_group="SO(3)",
                ) 

optimizer = optim.Adam(model.parameters(), lr=1e-4)
c_loss = torch.nn.MSELoss()

for epoch in range(10):
    print(f"Current Epoch:{epoch}")
    model.train()
    running_loss = 0.0
    print("Running Training Batches")
    for batch in tqdm(train_loader):
        args= batch.to_dict()
        y_true = args['y'].float()
        y = model(args['z'], args['pos'], args['batch'])
        optimizer.zero_grad()
        y_pred = y[3].requires_grad_()
        loss = c_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_train_loss = running_loss / len(train_loader)

    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        print("Running Validation Batches")
        for batch in tqdm(val_loader):
            args = batch.to_dict()
            y_true = args['y'].float()
            y = model(args['z'], args['pos'], args['batch'])
            y_pred = y[3].requires_grad_()
            val_loss = c_loss(y_pred, y_true)
            val_running_loss += val_loss.item()

    epoch_val_loss = val_running_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{10}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

print("Finished")


torch.save(model.state_dict(), 'model.pt')

