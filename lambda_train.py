import torch.nn as nn
from torch import Tensor, cat, ones_like, device
import torch
import torch.optim as optim
from torch.autograd import grad
import gc
import random
import math
import wandb
import terrace as ter
from torchmdnet.models.tensornet import TensorNet
from tqdm import tqdm
from datasets.bigbind_solv import BigBindSolvDataset
from torchmdnet.models.utils import scatter



#Parameters to change
WANDB_PROJ_NAME: str = "ML Implicit Solvent"
WANDB_GRAPH_NAME: str = ''
SLURMM_OUTPUT_TITLE_NAME: str = ''

HIDDEN_CHANNELS: int = 128
TENSOR_NET_LAYERS: int = 2
LAMBDA_INTEGRATION_LAYERS: int = 2
MAX_NUM_NEIGHBORS:int = 100
BATCH_SIZE: int = 4
WEIGHT_DECAY: float = 0.3
CLIP: float = 1
INITIAL_LEARNING_RATE: float = 1e-4
MINIMUM_LR: float = 1e-15
EPOCHS: int = 200






class ISAI(nn.Module):

    def __init__(self, hidden_channels, 
                 tensor_net_layers, 
                 lambda_integration_layers,
                 max_num_neighbors,
                 derivative

    ):
        super(ISAI, self).__init__()
        self.TensorNet = TensorNet(hidden_channels=hidden_channels,
                                    num_layers=tensor_net_layers,
                                    max_num_neighbors=max_num_neighbors,
                                    equivariance_invariance_group= "SO(3)")

        self.ReduceModel = LambdaScalar(hidden_channels=hidden_channels,
                                        integration_layers=lambda_integration_layers)

        self.derivative = derivative

    def derivativeCalc(self, constraint, y, pos, lambda_sterics, lambda_electrostatics):
        if constraint:
            y = y*(lambda_electrostatics+lambda_sterics)
        grad_outputs = ones_like(y)
        dy = grad([y],
                  [pos],
                  grad_outputs=grad_outputs,
                  create_graph=True,
                  retain_graph=True)[0]
        assert dy is not None, "Autograd returned None for the force prediction."
        dSterics = grad([y],
                        [lambda_sterics],
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True)[0]
        assert dSterics is not None, "Autograd returned None for the force prediction."
        dElectrostatics = grad([y], 
                        [lambda_electrostatics], 
                        grad_outputs=grad_outputs, 
                        create_graph=True,
                        retain_graph=True)[0]
        assert dElectrostatics is not None, "Autograd returned None for the force prediction."

        return -dy, dSterics, dElectrostatics 

        
    def forward(self, 
                z: Tensor,
                pos: Tensor, 
                batch: Tensor,
                lambda_electrostatics,
                lambda_sterics):
        
        if self.derivative: 
            pos.requires_grad_()
        
        y,_,z,pos,batch = self.TensorNet(z,pos, batch)

        y = self.ReduceModel(y, lambda_sterics, lambda_electrostatics, batch)



        if self.derivative:
            negdy, dSterics, dElectrostatics = self.derivativeCalc(contraint = True, y=y, pos=pos,
                                lamba_electrostatics= lambda_electrostatics, 
                                lambda_sterics=lambda_sterics)
            return negdy, dSterics, dElectrostatics
        else:
            return y

        
class LambdaScalar(nn.Module):
    
    def __init__(self, integration_layers, hidden_channels):
        super(LambdaScalar, self).__init__()

        self.layers = nn.Sequential()

        for _ in range(integration_layers):
            self.layers.append(nn.Linear(hidden_channels + 2, hidden_channels + 2))
            self.layers.append(nn.BatchNorm2d(hidden_channels+2, hidden_channels+2))
            self.layers.append(nn.ReLU)
        self.layers.append(nn.Linear(hidden_channels + 2, 1))

    def forward(self, y, lambda_sterics, lambda_electrostatics, batch):
        
        #Applies the MLP on the Lambdas and the values of the TensorNet
        lambda_electrostatics, lambda_sterics = lambda_electrostatics[batch].view(-1, 1), lambda_sterics[batch].view(-1, 1)
        y = cat((y, lambda_electrostatics, lambda_sterics),dim=1)

        y = self.layers(y)
        return scatter(y, batch, dim=0, dim_size=0)
    

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
    device = device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = BigBindSolvDataset("train")
    train_loader=ter.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_dataset = BigBindSolvDataset("val")
    val_loader = ter.DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=True)
    test_dataset = BigBindSolvDataset("test")
    test_loader = ter.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True)

    model = ISAI(hidden_channels=HIDDEN_CHANNELS,
                 tensor_net_layers=TENSOR_NET_LAYERS,
                 lambda_integration_layers= LAMBDA_INTEGRATION_LAYERS,
                 max_num_neighbors=MAX_NUM_NEIGHBORS,
                 derivative=True)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.1, 
                                                    patience=10, threshold=1e3, threshold_mode='abs',
                                                    cooldown=0, min_lr=MINIMUM_LR, eps=1e-20)
    model.to(device)
    model.train()
    #If changing LR twice does not decrease.
    early_stopper = EarlyStopper(patience=10, min_delta=1)


    for epoch in range(EPOCHS):
        print(f"Current Epoch: {epoch}")
        print("Training Now: ")
        running_loss = 0.0
        running_lossdy = 0.0
        running_loss_elec = 0.0
        running_loss_ster = 0.0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            lambdaElecGrad = batch.lambda_electrostatics.requires_grad_().to(device)
            lambdaStericsGrad = batch.lambda_sterics.requires_grad_().to(device)
            lambdaElecTrue = batch.electrostatics_derivative.requires_grad_().to(device)
            lambdaStericsTrue = batch.sterics_derivative.requires_grad_().to(device)
            y_true = batch.forces.requires_grad_().to(device)

            negdy, dSterics, dElectrostatics = model(z=batch.atomic_numbers.to(device),
                                                    pos=batch.positions.to(device),
                                                    batch=batch.batch.to(device),
                                                    lambda_electrostatics=lambdaElecGrad,
                                                    lambda_sterics=lambdaStericsGrad,)
            lossdy = criterion(negdy, y_true) 
            loss_elec = criterion(dElectrostatics, lambdaStericsTrue) 
            loss_ster = criterion(dSterics, lambdaElecTrue)
            loss = lossdy + loss_elec + loss_ster

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            scheduler.step(loss.item())
            wandb.log({WANDB_GRAPH_NAME: loss.item()})

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
        
        print(f"Training Loss: {train_loss}")
        print(f"Training Lossdy: {train_lossdy}")
        print(f"Training Loss_elec: {train_loss_elec}")
        print(f"Training Loss_ster: {train_loss_ster}")

    name = random.randint(-math.inf, math.inf)
    print(f"model_name: name")    
    torch.save(model.state_dict(), f'{name}.pt')            

if __name__ == "__main__":
    wandb.init(project = WANDB_PROJ_NAME)
    print(SLURMM_OUTPUT_TITLE_NAME)

    train()
