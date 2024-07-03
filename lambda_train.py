
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
import os
from typing import Optional



#Parameters to change

#SiLU is the only that works for autograd lambda values; works better with AMSGrad turned on
OUTPUT_COMMENT = "Added Charges"

#Interpretability Variables
LOSS_VALIDATION: bool = False # PRINTS LOSS PER LABEL IF ABOVE MAX_LOSS - This is broken lmao
MAX_LOSS: int = 1000000 # This is dependent on LOSS_VALIDATION
DISABLE_LAMBDA: bool = False # Disables lambda calculations - LOSS VALIDATION DOES NOT WORK WITH THIS DISABLED
ONE_FILTER = True #Prints r2 masked against lambdavalues of one


SAVE_RATE = 3 #How often to save the model
SAVE_PATH = '/users/r/d/rdey/'


VALIDATION = True #Edit: I fixed this :) -- DOES NOT WORK - I was slightly lazy to add a single if statement
BATCH_DISABLER_INT = -1 #-1 disables it; Limits the number of batches for testing purposes
SHUFFLE = True
HIDDEN_CHANNELS: int = 128
TENSOR_NET_LAYERS: int = 3 #Amount of layers in the TensorNet
LAMBDA_INTEGRATION_LAYERS: int = 18 #Amount of layers in the MLP
MAX_NUM_NEIGHBORS: int = 128
BATCH_SIZE: int = 4
WEIGHT_DECAY: float = 0.01
CLIP: float = -1 #Gradient Clipping; -1 Disables 
INITIAL_LEARNING_RATE: float = 1e-4 #1e-4 best for dy; 1e-5 best for lambdas 
SCHEDULER: bool = False #Activates the scheduler
MINIMUM_LR: float = 1e-15 #Dependent on scheduler
PATIENCE = 1 #Dependent on scheduler
EPOCHS: int = 1000
EARLY_STOP = False #Activates Earlystopper 

LOSS_COEFFICIENT_DY = 8
LOSS_COEFFICIENT_ELECTROSTATICS = 3
LOSS_COEFFICIENT_STERICS = 2


#Wandb Variables
CONNECT_WANDB: bool = True
WANDB_PROJ_NAME: str = "ML Implicit Solvent"
WANDB_GRAPH_NAME: str = f'5.0'
SLURMM_OUTPUT_TITLE_NAME: str = WANDB_GRAPH_NAME

TESTING_MASTERCLASS = False # Overfitting; Disables and changes many of the above parameters for testing purposes




if TESTING_MASTERCLASS:
    VALIDATION = False
    SAVE_RATE = 10000000
    SHUFFLE = False
    CONNECT_WANDB = False
    BATCH_DISABLER_INT = 1



#Main Ensemble NN class 
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

        dSterics = grad([y],
                        [lambda_sterics],
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True)[0]
        assert dSterics is not None, "Autograd returned None for the force prediction."
        dy = grad([y],
                  [pos],
                  grad_outputs=grad_outputs,
                  create_graph=True,
                  retain_graph=True)[0]
        assert dy is not None, "Autograd returned None for the force prediction."

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
                q: Tensor, 
                batch: Tensor,
                lambda_electrostatics,
                lambda_sterics,
                disable_lambdas):
        
        if self.derivative: 
            pos.requires_grad_()

        #lambda_electrostatics = [1 if val is None else val for val in lambda_electrostatics]
        #lambda_sterics = [1 if val is None else val for val in lambda_sterics]
            
        #Accounts charges for lambda_electrostatics, allows model to focus more on the nonpolar components
        q = q*torch.sqrt(lambda_electrostatics[batch].detach())

        y,_,z,pos,batch = self.TensorNet(z=z,pos=pos,q=q,batch=batch)
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
            layers.append(nn.ELU())
        layers.append(nn.Linear(hidden_channels + 2, 1))
        
        self.layers = nn.Sequential(*layers)
        
        self.reset_parameters()
    #model learns better with xavier with the lambdas
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
            y = torch.cat((y, lambda_electrostatics, lambda_sterics),dim=1)

        y = self.layers(y)
        is_capturing = y.is_cuda and is_current_stream_capturing()
        if not y.is_cuda or not is_capturing:
            self.dim_size = int(batch.max().item() + 1)
        
        return scatter(y, batch, dim=0, dim_size=self.dim_size)
    
#Early Stopper to stop the NN if not learning
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


#Main training definition
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = BigBindSolvDataset("train", frame_index=7)
    train_loader=ter.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=SHUFFLE)
    print(len(train_dataset))
    val_dataset = BigBindSolvDataset("val", frame_index=1)
    val_loader = ter.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ISAI(hidden_channels=HIDDEN_CHANNELS,
                 tensor_net_layers=TENSOR_NET_LAYERS,
                 lambda_integration_layers= LAMBDA_INTEGRATION_LAYERS,
                 max_num_neighbors=MAX_NUM_NEIGHBORS,
                 derivative=True)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay = WEIGHT_DECAY, amsgrad=True,)

    if(SCHEDULER):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.1, 
                                                    patience=PATIENCE, threshold=1e-3, threshold_mode='abs',
                                                    cooldown=0, min_lr=MINIMUM_LR, eps=1e-10)
    model.to(device)

    #If changing LR twice does not decrease.
    early_stopper = EarlyStopper(patience=10, min_delta=1)
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(EPOCHS):
        count = 0
        model.train()
        print(f"Current Epoch: {epoch}")
        print("Training Now: ")

        running_loss = 0.0
        running_lossdy = 0.0
        running_loss_elec = 0.0
        running_loss_ster = 0.0

        true_ys = []
        predicted_ys = []
        filter_predicted = []
        filter_true = []
        l_sterics_true = []
        l_elec_true = []
        l_sterics_predicted = []
        l_elec_predicted = []

        for batch in train_loader:
            
            optimizer.zero_grad()
            lambdaElec = batch.lambda_electrostatics.requires_grad_().to(device)
            lambdaSterics = batch.lambda_sterics.requires_grad_().to(device)

            lambdaElecTrue = batch.electrostatics_derivative.to(device)
            lambdaStericsTrue = batch.sterics_derivative.to(device)

            y_true = batch.forces.to(device)

            

            true_ys.append(y_true.detach().cpu().numpy().flatten())
            l_sterics_true.append(lambdaStericsTrue.cpu().view(-1,1).numpy())
            l_elec_true.append(lambdaElecTrue.cpu().view(-1,1).numpy())
            

            
            negdy, dSterics, dElectrostatics = model(z=batch.atomic_numbers.to(device),
                                                    pos=batch.positions.to(device),
                                                    q = batch.charges.to(device),
                                                    batch=batch.batch.to(device),
                                                    lambda_electrostatics=lambdaElec,
                                                    lambda_sterics=lambdaSterics,
            
                                                    disable_lambdas = DISABLE_LAMBDA)

            predicted_ys.append(negdy.detach().cpu().numpy().flatten())
            l_sterics_predicted.append(dSterics.detach().cpu().view(-1,1).numpy())
            l_elec_predicted.append(dElectrostatics.detach().cpu().view(-1,1).numpy())
            
            if not DISABLE_LAMBDA:

                #Filter for ones for lambda values
                if ONE_FILTER: 
                    mask = (lambdaElec.cpu() == 1) & (lambdaSterics.cpu() == 1)
                    mask = mask[batch.batch]
                    filter_true.append(y_true[mask].detach().cpu().numpy().flatten())
                    filter_predicted.append(negdy[mask].detach().cpu().numpy().flatten())

                lossdy = criterion(negdy, y_true)
                loss_elec = criterion(dElectrostatics,lambdaElecTrue) 
                loss_ster = criterion(dSterics, lambdaStericsTrue)
                loss = lossdy*LOSS_COEFFICIENT_DY + loss_elec*LOSS_COEFFICIENT_ELECTROSTATICS + loss_ster*LOSS_COEFFICIENT_STERICS
                
            loss.backward()

            if CLIP != -1:
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
            #del lambdaElecGrad, lambdaStericsGrad, lambdaElecTrue, lambdaStericsTrue, y_true
            #del negdy, dSterics, dElectrostatics, loss, lossdy 
            
           # if not DISABLE_LAMBDA:
                #del loss_elec, loss_ster

            #torch.cuda.empty_cache()
            gc.collect()

            if BATCH_DISABLER_INT != -1:
                if(count == BATCH_DISABLER_INT):
                    break
                else:
                    count +=1

        
        #Accuracy Calculations for interpretability
        filter_r2 = 0
        true_ys = np.concatenate(true_ys)
        predicted_ys = np.concatenate(predicted_ys)
        l_elec_predicted = np.concatenate(l_elec_predicted)
        l_elec_true = np.concatenate(l_elec_true)
        l_sterics_predicted = np.concatenate(l_sterics_predicted)
        l_sterics_true = np.concatenate(l_sterics_true)

        if ONE_FILTER and not DISABLE_LAMBDA:

            filter_true = np.concatenate(filter_true)
            filter_predicted = np.concatenate(filter_predicted)
            filter_r2 = r2_score(filter_true, filter_predicted)

        train_r2_dy = r2_score(true_ys, predicted_ys)
        train_r2_sterics = r2_score(l_sterics_true, l_sterics_predicted)
        train_r2_elec = r2_score(l_elec_true, l_elec_predicted)

        print(f"R2_dy: {train_r2_dy}")
        print(f"R2_elec: {train_r2_elec}")
        print(f"R2_ster: {train_r2_sterics}")
        print(f"FilterR2: {filter_r2}")

        if BATCH_DISABLER_INT == -1:
            train_loss = running_loss / len(train_loader)
            train_lossdy = running_lossdy / len(train_loader)
            train_loss_elec = running_loss_elec / len(train_loader)
            train_loss_ster = running_loss_ster / len(train_loader)
        else: 
            train_loss = running_loss / BATCH_DISABLER_INT
            train_lossdy = running_lossdy / BATCH_DISABLER_INT
            train_loss_elec = running_loss_elec / BATCH_DISABLER_INT
            train_loss_ster = running_loss_ster / BATCH_DISABLER_INT 
        
        print(f"Training Loss: {train_loss}")
        print(f"Training Lossdy: {train_lossdy}")
        print(f"Training Loss_elec: {train_loss_elec}")
        print(f"Training Loss_ster: {train_loss_ster}")
        

        if not VALIDATION:
            continue

        print("Validation Now:")
        model.eval()
        running_loss = 0.0
        running_lossdy = 0.0
        running_loss_elec = 0.0
        running_loss_ster = 0.0
        
        val_true_ys = []
        val_predicted_ys = []
        filter_val_true = []
        filter_val_predicted = []


        for batch in val_loader:
            optimizer.zero_grad()
            lambdaElecGrad = batch.lambda_electrostatics.requires_grad_().to(device)
            lambdaStericsGrad = batch.lambda_sterics.requires_grad_().to(device)
            lambdaElecTrue = batch.electrostatics_derivative.to(device)
            lambdaStericsTrue = batch.sterics_derivative.to(device)
            y_true = batch.forces.to(device)
            val_true_ys.append(y_true.detach().cpu().numpy().flatten())

            negdy, dSterics, dElectrostatics = model(z=batch.atomic_numbers.to(device),
                                                    pos=batch.positions.to(device),
                                                    q = batch.charges.to(device),
                                                    batch=batch.batch.to(device),
                                                    lambda_electrostatics=lambdaElecGrad,
                                                    lambda_sterics=lambdaStericsGrad,
            
                                                    disable_lambdas = DISABLE_LAMBDA)
            
            
            val_predicted_ys.append(negdy.detach().cpu().numpy().flatten())


            
            if not DISABLE_LAMBDA: 
                lossdy = criterion(negdy, y_true)
                loss_elec = criterion(dElectrostatics, lambdaStericsTrue) 
                loss_ster = criterion(dSterics, lambdaElecTrue)
                loss = lossdy*LOSS_COEFFICIENT_DY + loss_elec*LOSS_COEFFICIENT_ELECTROSTATICS + loss_ster*LOSS_COEFFICIENT_STERICS

            running_loss += loss.item()
            running_lossdy += lossdy.item()

            if not DISABLE_LAMBDA: 

                if ONE_FILTER: 
                    mask = (lambdaElecGrad.cpu() == 1) & (lambdaStericsGrad.cpu() == 1)
                    mask = mask[batch.batch]
                    filter_val_true.append(y_true[mask].detach().cpu().numpy().flatten())
                    filter_val_predicted.append(negdy[mask].detach().cpu().numpy().flatten())


                running_loss_elec += loss_elec.item()
                running_loss_ster += loss_ster.item()

            # Clear cache and garbage collect to free memory
            del lambdaElecGrad, lambdaStericsGrad, lambdaElecTrue, lambdaStericsTrue, y_true
            del negdy, dSterics, dElectrostatics, loss, lossdy 
            
            if not DISABLE_LAMBDA:
                del loss_elec, loss_ster

            torch.cuda.empty_cache()
            gc.collect()

        if ONE_FILTER and not DISABLE_LAMBDA:

            filter_true = np.concatenate(filter_true)
            filter_predicted = np.concatenate(filter_predicted)
            filter_val_r2 = r2_score(filter_val_true, filter_val_predicted)
            
        val_loss = running_loss / len(val_loader)
        val_lossdy = running_lossdy / len(val_loader)
        val_loss_elec = running_loss_elec / len(val_loader)
        val_loss_ster = running_loss_ster / len(val_loader)

        true_ys = np.concatenate(val_true_ys)
        predicted_ys = np.concatenate(val_predicted_ys)

        val_r2 = r2_score(true_ys, predicted_ys)
        print(f"R2: {val_r2}")
        print(f"Val_Filter_R2: {filter_val_r2}")

        print(f"Validation Loss: {val_loss}")
        print(f"Validation Lossdy: {val_lossdy}")
        print(f"Validation Loss_elec: {val_loss_elec}")
        print(f"Validation Loss_ster: {val_loss_ster}")
        for param_group in optimizer.param_groups:
            print(f"Learning rate is {param_group['lr']}")
            if(CONNECT_WANDB):
                wandb.log({f"{WANDB_GRAPH_NAME} LR": param_group['lr']})

        if(EPOCHS % SAVE_RATE == 0):
            name = str(random.randint(0, 100000000000))  # This ensures the range is correctly set with integers
            print(f"model_name: {name}")
            torch.save(model.state_dict(), f=os.path.join(SAVE_PATH, f'{name}.pt'))
            if(CONNECT_WANDB):
                wandb.log({f"{WANDB_GRAPH_NAME} LOGGED MODEL": name})            
        
        if(CONNECT_WANDB):
            wandb.log({f"{WANDB_GRAPH_NAME} TRAINING LOSS AGGREGATE": train_loss})
            wandb.log({f"{WANDB_GRAPH_NAME} TRAINING LOSS FORCES": train_lossdy})
            wandb.log({f"{WANDB_GRAPH_NAME} TRAINING LOSS STERICS": train_loss_ster})
            wandb.log({f"{WANDB_GRAPH_NAME} TRAINING LOSS ELECTROSTATICS": train_loss_elec})
            wandb.log({f"{WANDB_GRAPH_NAME} TRAINING R2_DY": train_r2_dy})
            wandb.log({f"{WANDB_GRAPH_NAME} TRAINING R2_STERICS": train_r2_elec})
            wandb.log({f"{WANDB_GRAPH_NAME} TRAINING R2_ELECTROSTATICS": train_r2_sterics})
            wandb.log({f"{WANDB_GRAPH_NAME} VALIDATION LOSS AGGREGATE": val_loss})
            wandb.log({f"{WANDB_GRAPH_NAME} VALIDATION LOSS FORCES": val_lossdy})
            wandb.log({f"{WANDB_GRAPH_NAME} VALIDATION LOSS STERICS": val_loss_ster})
            wandb.log({f"{WANDB_GRAPH_NAME} VALIDATION LOSS ELECTROSTATICS": val_loss_elec})
            wandb.log({f"{WANDB_GRAPH_NAME} VALIDATION R2_DY": val_r2})
            wandb.log({f"{WANDB_GRAPH_NAME} TRAINING FILTERED R2": filter_r2})
            wandb.log({f"{WANDB_GRAPH_NAME} VALIDATION FILTERED R2": filter_val_r2})
            

        #Future processing for future epochs in training
        if(SCHEDULER):
                scheduler.step(val_loss)     
        if EARLY_STOP: 
            if(early_stopper.early_stop(val_loss)):
                print(f"Early Stopped, loss not decreasing")
    
if __name__ == "__main__":
    print(OUTPUT_COMMENT)
    if(CONNECT_WANDB):
        print("hi")
        run = wandb.init(
            project = WANDB_PROJ_NAME,
            config = {
                "Hidden Channels": HIDDEN_CHANNELS,
                "Tensor Net Layers": TENSOR_NET_LAYERS,
                "Lambda Integration Layers": LAMBDA_INTEGRATION_LAYERS,
                "Max Number Neighbors": MAX_NUM_NEIGHBORS,
                "Batch Size": BATCH_SIZE,
                "Weight Decay": WEIGHT_DECAY,
                "Gradient Clipping": CLIP,
                "Initial LR": INITIAL_LEARNING_RATE,
                "Scheduler?": SCHEDULER,
                "Num of Epochs": EPOCHS,
                "Early Stop": EARLY_STOP,
                "Loss Coefficient DY": LOSS_COEFFICIENT_DY,
                "Loss Coefficient Electrostatics": LOSS_COEFFICIENT_ELECTROSTATICS,
                "Loss Coefficient Sterics": LOSS_COEFFICIENT_STERICS,
                }
            )
        
    print(SLURMM_OUTPUT_TITLE_NAME)
    train()
