import os
from tqdm import tqdm
from config import CONFIG
import terrace as ter
from datasets.bigbind_solv import BigBindSolvDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchmdnet.models.model import TorchMD_Net
from torchmdnet.models.tensornet import TensorNet
from torchmdnet.models.output_modules import Scalar 
import gc
from datasets.TensorNetLambda import LambdaLoss
import random 
import math
import wandb

wandb.init(
    project = "ML Implicit Solvent"
)



print("LAYERS ADDED - WANDB LOSS - BATCHNORM - LEARNING RATE: 1E-4 + SCHEDULER - BATCH SIZE-4 - GRAD CLIPPED - 1")
# Using Class to introduce early stopping criterion to avoid overfitting of network
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initializing the dataset and dataloader. Using BigBindSolvent data as reference.
    train_dataset = BigBindSolvDataset("train")
    train_loader=ter.DataLoader(train_dataset, batch_size = 4, shuffle=True)
    val_dataset = BigBindSolvDataset("val")
    val_loader = ter.DataLoader(val_dataset, batch_size = 4, shuffle=True)
    test_dataset = BigBindSolvDataset("test")
    test_loader = ter.DataLoader(test_dataset, batch_size = 4, shuffle=True)


    # Initializing the modified TensorNet model, loss function, and optimizer
    model = TorchMD_Net(
                representation_model= 
                    TensorNet(
                        hidden_channels=128,
                        num_layers=2,
                        num_rbf=32,
                        rbf_type="expnorm",
                        trainable_rbf=True,
                        activation='ssp',
                        equivariance_invariance_group='O(3)',
                        cutoff_lower=0.0, # Default 
                        cutoff_upper=4.5,  # Default  
                        max_num_neighbors= 120 #Default is 64    
                    ), 
                output_model = 
                    Scalar(hidden_channels=128,
                           num_layers = 2), 
                derivative = True,
                
    )
    
       
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                 patience=10, threshold=1e3, threshold_mode='abs',
                                                 cooldown=0, min_lr=1e-15, eps=1e-20)
    
    model.to(device)
    

    # Using early_stopper class
    early_stopper = EarlyStopper(patience=5, min_delta=0.001)
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(200):
        model.train()
        print(f"Current Epoch: {epoch}")
        print("Training Now: ")
        running_loss = 0.0
        running_lossdy = 0.0
        running_loss_elec = 0.0
        running_loss_ster = 0.0        
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            lambdaElecGrad = batch.lambda_electrostatics.requires_grad_().to(device)
            lambdaStericsGrad = batch.lambda_sterics.requires_grad_().to(device)
            lambdaElecTrue = batch.electrostatics_derivative.requires_grad_().to(device)
            lambdaStericsTrue = batch.sterics_derivative.requires_grad_().to(device)
            y_true = batch.forces.requires_grad_().to(device)

            y, negdy, dSterics, dElectrostatics = model(
                z=batch.atomic_numbers.to(device),
                pos=batch.positions.to(device),
                batch=batch.batch.to(device),
                lambda_electrostatics=lambdaElecGrad,
                lambda_sterics=lambdaStericsGrad,
                box=None,
                Training = True
            )

            lossdy = criterion(negdy, y_true) 
            loss_elec = criterion(dElectrostatics, lambdaStericsTrue) 
            loss_ster = criterion(dSterics, lambdaElecTrue)
            loss = lossdy + loss_elec + loss_ster
                    
            loss.backward()
            clip = 1
            torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
            optimizer.step()
            scheduler.step(loss.item())
            wandb.log({"V2 - loss(E-4 + SCHEDULER) - BS4": loss.item()})

            running_loss += loss.item()
            running_lossdy += lossdy.item()
            running_loss_elec += loss_elec.item()
            running_loss_ster += loss_ster.item()

            # Clear cache and garbage collect to free memory
            del lambdaElecGrad, lambdaStericsGrad, lambdaElecTrue, lambdaStericsTrue, y_true
            del y, negdy, dSterics, dElectrostatics, loss, lossdy, loss_elec, loss_ster
            torch.cuda.empty_cache()
            gc.collect()
        
        train_loss = running_loss / len(train_loader)
        train_lossdy = running_lossdy / len(train_loader)
        train_loss_elec = running_loss_elec / len(train_loader)
        train_loss_ster = running_loss_ster / len(train_loader)
        train_losses.append(train_loss)
        
        print(f"Training Loss: {train_loss}")
        print(f"Training Lossdy: {train_lossdy}")
        print(f"Training Loss_elec: {train_loss_elec}")
        print(f"Training Loss_ster: {train_loss_ster}")

    name = random.randint(-math.inf, math.inf)
    print(f"model_name: name")    
    torch.save(model.state_dict(), f'{name}.pt')
if __name__ == "__main__":
    train()


