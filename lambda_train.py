#AI model of Solvation using TorchMDNet TensorNet
#Need a better name than ISAI
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
from torchmdnet.extensions import is_current_stream_capturing
import numpy as np
from sklearn.metrics import r2_score
from torch.nn.utils.rnn import pad_sequence



#Parameters to change
LOSS_VALIDATION: bool = False # PRINTS LOSS PER LABEL IF ABOVE MAX_LOSS
MAX_LOSS: int = 1000000
DISABLE_LAMBDA: bool = False # LOSS VALIDATION DOES NOT WORK WITH THIS DISABLED

HIDDEN_CHANNELS: int = 128
TENSOR_NET_LAYERS: int = 2
LAMBDA_INTEGRATION_LAYERS: int = 3
MAX_NUM_NEIGHBORS: int = 120

BATCH_SIZE: int = 4
WEIGHT_DECAY: float = 1e-2
CLIP: float = 1
INITIAL_LEARNING_RATE: float = 1e-4
SCHEDULER: bool = True
MINIMUM_LR: float = 1e-15
EPOCHS: int = 1000
EARLY_STOP = True

CONNECT_WANDB: bool = False
WANDB_PROJ_NAME: str = "ML Implicit Solvent"
WANDB_GRAPH_NAME: str = f'2.0 STATIC FRAME IDX w/ VALIDATION- SCHEDULER {INITIAL_LEARNING_RATE} BS{BATCH_SIZE} GS{CLIP}'
SLURMM_OUTPUT_TITLE_NAME: str = WANDB_GRAPH_NAME


VALIDATION = True


#Main class 
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
                lambda_sterics,
                disable_lambdas):
        
        if self.derivative: 
            pos.requires_grad_()
        
        y,_,z,pos,batch = self.TensorNet(z,pos, batch)

        y = self.ReduceModel(y, lambda_sterics, lambda_electrostatics, disable_lambdas, batch)

        if self.derivative:
            negdy, dSterics, dElectrostatics = self.derivativeCalc(constraint = True, y=y, pos=pos,
                                lambda_electrostatics= lambda_electrostatics, 
                                lambda_sterics=lambda_sterics)
            return negdy, dSterics, dElectrostatics
        else:
            return y

#Hidden Channel Convergence Model         
class LambdaScalar(nn.Module):
    
    def __init__(self, integration_layers, hidden_channels):
        super(LambdaScalar, self).__init__()

        layers = []
        for _ in range(integration_layers):
            layers.append(nn.Linear(hidden_channels + 2, hidden_channels + 2))
            layers.append(nn.BatchNorm1d(hidden_channels + 2))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_channels + 2, 1))
        
        self.layers = nn.Sequential(*layers)
        
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, y, lambda_sterics, lambda_electrostatics, disable_lambdas, batch):
        
        #Applies the MLP on the Lambdas and the values of the TensorNet
        if not disable_lambdas: 
            lambda_electrostatics, lambda_sterics = lambda_electrostatics[batch].view(-1, 1), lambda_sterics[batch].view(-1, 1)
            y = cat((y, lambda_electrostatics, lambda_sterics),dim=1)

        y = self.layers(y)
        is_capturing = y.is_cuda and is_current_stream_capturing()
        if not y.is_cuda or not is_capturing:
            self.dim_size = int(batch.max().item() + 1)
        
        return scatter(y, batch, dim=0, dim_size=self.dim_size)
    

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
    train_dataset = BigBindSolvDataset("train", frame_index=0)
    train_loader=ter.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_dataset = BigBindSolvDataset("val", frame_index=0)
    val_loader = ter.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ISAI(hidden_channels=HIDDEN_CHANNELS,
                 tensor_net_layers=TENSOR_NET_LAYERS,
                 lambda_integration_layers= LAMBDA_INTEGRATION_LAYERS,
                 max_num_neighbors=MAX_NUM_NEIGHBORS,
                 derivative=True)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay = WEIGHT_DECAY)

    if(SCHEDULER):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.1, 
                                                    patience=10, threshold=1e-3, threshold_mode='abs',
                                                    cooldown=0, min_lr=MINIMUM_LR, eps=1e-20)
    model.to(device)
    model.train()

    #If changing LR twice does not decrease.
    early_stopper = EarlyStopper(patience=10, min_delta=1)


    for epoch in range(EPOCHS):
        print(f"Current Epoch: {epoch}")
        print("Training Now: ")
        for param_group in optimizer.param_groups:
            print(f"Learning rate is {param_group['lr']}")
        running_loss = 0.0
        running_lossdy = 0.0
        running_loss_elec = 0.0
        running_loss_ster = 0.0

        true_ys = []
        predicted_ys = []
        counter = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            lambdaElecGrad = batch.lambda_electrostatics.requires_grad_().to(device)
            lambdaStericsGrad = batch.lambda_sterics.requires_grad_().to(device)
            lambdaElecTrue = batch.electrostatics_derivative.to(device)
            lambdaStericsTrue = batch.sterics_derivative.to(device)
            y_true = batch.forces.to(device)

            true_ys.append(y_true.detach().numpy().flatten())
            
            negdy, dSterics, dElectrostatics = model(z=batch.atomic_numbers.to(device),
                                                    pos=batch.positions.to(device),
                                                    batch=batch.batch.to(device),
                                                    lambda_electrostatics=lambdaElecGrad,
                                                    lambda_sterics=lambdaStericsGrad,
            
                                                    disable_lambdas = DISABLE_LAMBDA)
            predicted_ys.append(negdy.detach().numpy().flatten())
            
            if not DISABLE_LAMBDA: 
                lossdy = criterion(negdy, y_true)
                loss_elec = criterion(dElectrostatics, lambdaStericsTrue) 
                loss_ster = criterion(dSterics, lambdaElecTrue)
                loss = lossdy + loss_elec + loss_ster
                

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()



            running_loss += loss.item()
            running_lossdy += lossdy.item()
            if not DISABLE_LAMBDA: 
                running_loss_elec += loss_elec.item()
                running_loss_ster += loss_ster.item()

            if(LOSS_VALIDATION and not DISABLE_LAMBDA):
                if(loss.item() > MAX_LOSS):
                    print(f"negdy: {torch.max(negdy)}")
                    print(f"y_true: {torch.max(y_true)}")
                    print(f"dSterics: {torch.max(dSterics)}")
                    print(f"TrueSterics: {torch.max(lambdaStericsTrue)}")
                    print(f"dElectrostatics: {torch.max(dElectrostatics)}")
                    print(f"TrueElectrostatics: {torch.max(lambdaElecTrue)}")
       
            # Clear cache and garbage collect to free memory
            del lambdaElecGrad, lambdaStericsGrad, lambdaElecTrue, lambdaStericsTrue, y_true
            del negdy, dSterics, dElectrostatics, loss, lossdy 
            
            if not DISABLE_LAMBDA:
                del loss_elec, loss_ster

            torch.cuda.empty_cache()
            gc.collect()

        
        
        true_ys = np.concatenate(true_ys)
        predicted_ys = np.concatenate(predicted_ys)

        train_r2 = r2_score(true_ys, predicted_ys)
        print(f"R2: {train_r2}")

        train_loss = running_loss / len(train_loader)
        train_lossdy = running_lossdy / len(train_loader)
        train_loss_elec = running_loss_elec / len(train_loader)
        train_loss_ster = running_loss_ster / len(train_loader)


        
        
        print(f"Training Loss: {train_loss}")
        print(f"Training Lossdy: {train_lossdy}")
        print(f"Training Loss_elec: {train_loss_elec}")
        print(f"Training Loss_ster: {train_loss_ster}")


        print("Validation Now:")
        model.eval()
        running_loss = 0.0
        running_lossdy = 0.0
        running_loss_elec = 0.0
        running_loss_ster = 0.0
        true_ys = []
        predicted_ys = []
        count = 0 
        for batch in val_loader:
            optimizer.zero_grad()
            lambdaElecGrad = batch.lambda_electrostatics.requires_grad_().to(device)
            lambdaStericsGrad = batch.lambda_sterics.requires_grad_().to(device)
            lambdaElecTrue = batch.electrostatics_derivative.to(device)
            lambdaStericsTrue = batch.sterics_derivative.to(device)
            y_true = batch.forces.to(device)

            true_ys.append(y_true.detach().numpy().flatten())

            negdy, dSterics, dElectrostatics = model(z=batch.atomic_numbers.to(device),
                                                    pos=batch.positions.to(device),
                                                    batch=batch.batch.to(device),
                                                    lambda_electrostatics=lambdaElecGrad,
                                                    lambda_sterics=lambdaStericsGrad,
            
                                                    disable_lambdas = DISABLE_LAMBDA)
            predicted_ys.append(negdy.detach().numpy().flatten())


            
            if not DISABLE_LAMBDA: 
                lossdy = criterion(negdy, y_true)
                loss_elec = criterion(dElectrostatics, lambdaStericsTrue) 
                loss_ster = criterion(dSterics, lambdaElecTrue)
                loss = lossdy + loss_elec + loss_ster

            if(CONNECT_WANDB):
                wandb.log({"Validation Loss": loss.item()})

            running_loss += loss.item()
            running_lossdy += lossdy.item()
            if not DISABLE_LAMBDA: 
                running_loss_elec += loss_elec.item()
                running_loss_ster += loss_ster.item()

            # Clear cache and garbage collect to free memory
            del lambdaElecGrad, lambdaStericsGrad, lambdaElecTrue, lambdaStericsTrue, y_true
            del negdy, dSterics, dElectrostatics, loss, lossdy 
            
            if not DISABLE_LAMBDA:
                del loss_elec, loss_ster

            torch.cuda.empty_cache()
            gc.collect()
            
        val_loss = running_loss / len(val_loader)
        val_lossdy = running_lossdy / len(val_loader)
        val_loss_elec = running_loss_elec / len(val_loader)
        val_loss_ster = running_loss_ster / len(val_loader)

        val_r2 = r2_score(true_ys, predicted_ys)
        print(f"R2: {val_r2}")

        print(f"Validation Loss: {val_loss}")
        print(f"Validation Lossdy: {val_lossdy}")
        print(f"Validation Loss_elec: {val_loss_elec}")
        print(f"Validation Loss_ster: {val_loss_ster}")

        
        if(CONNECT_WANDB):
            wandb.log({f"{WANDB_GRAPH_NAME} TRAINING LOSS": train_loss})
            wandb.log({f"{WANDB_GRAPH_NAME} TRAINING R2": train_r2})
            wandb.log({f"{WANDB_GRAPH_NAME} VALIDATION LOSS": val_loss})
            wandb.log({f"{WANDB_GRAPH_NAME} VALIDATION R2": val_r2})

        if(SCHEDULER):
                scheduler.step(val_loss)     
        if EARLY_STOP: 
            if(early_stopper.early_stop(val_loss)):
                print(f"Early Stopped, loss not decreasing")
                break
        



    name = random.randint(-1e6, 1e6)
    print(f"model_name: {name}")    
    torch.save(model.state_dict(), f'{name}.pt')


           

if __name__ == "__main__":
    if(CONNECT_WANDB):
        print("hi")
        wandb.init(project = WANDB_PROJ_NAME)
    print(SLURMM_OUTPUT_TITLE_NAME)

    train()
