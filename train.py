from tqdm import tqdm
from config import CONFIG
from datasets.bigbind_solv import BigBindSolvDataset
import terrace as ter
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from tqdm import tqdm
import numpy as np
from datasets.TensorNetLambda import LambdaNet, LambdaLoss

def train():
    train_dataset = BigBindSolvDataset("train")
    train_loader = ter.DataLoader(train_dataset,
                                    batch_size=CONFIG.batch_size,
                                    shuffle=True)
    val_dataset = BigBindSolvDataset("val")
    val_loader = ter.DataLoader(val_dataset,
                                    batch_size=CONFIG.batch_size,
                                    shuffle=True)
    test_dataset = BigBindSolvDataset("test")
    test_loader = ter.DataLoader(test_dataset,
                                    batch_size=CONFIG.batch_size,
                                    shuffle=True)

    model = LambdaNet(hidden_channels=128, 
                    num_layers = 3, 
                    trainable_rbf = True, 
                    activation = 'silu', 
                    cutoff_upper = 10.0, 
                    equivariance_invariance_group="SO(3)",
                    max_num_neighbors= 100
                    ) 

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    c_loss = torch.nn.MSELoss()

    for epoch in range(10):
        print(f"Current Epoch:{epoch}")
        model.train()
        running_loss = 0.0
        print("Running Training Batches")
        for batches in tqdm(train_loader):
            y_true = batches.forces
            print(batches)
            y, neg_dy = model(z=batches.atomic_numbers, pos= batches.positions, batch= batches.batch, lambdaelec = batches.lambdaelec, lambdaster = batches.lambdaster)
            loss = LambdaLoss(y=y, c_loss = c_loss, batches=batches)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_train_loss = running_loss / len(train_loader)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            print("Running Validation Batches")
            for batches in tqdm(val_loader):
                y_true = batches.forces
                y, neg_dy = model(batches.atomic_numbers, batches.positions, batches.batch)
                y_pred = neg_dy.requires_grad_()
                val_loss = c_loss(y_pred, y_true)
                val_running_loss += val_loss.item()

        epoch_val_loss = val_running_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{10}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

    print("Finished")


    torch.save(model.state_dict(), 'model.pt')
    


if __name__ == "__main__":
    train()
