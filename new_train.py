import os
from tqdm import tqdm
from config import CONFIG
import terrace as ter
from datasets.bigbind_solv import BigBindSolvDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchmdnet.models.tensornet import TensorNet
from matplotlib.pyplot import plt
#from torchmdnet.models.model import create_model
import gc
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

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
    # Initializing the dataset and dataloader. Using BigBindSolvent data as reference.
    train_dataset = BigBindSolvDataset("train")
    train_loader=ter.DataLoader(train_dataset,batch_size=10,shuffle=True)
    val_dataset = BigBindSolvDataset("val")
    val_loader = ter.DataLoader(val_dataset, batch_size=10, shuffle=True)
    test_dataset = BigBindSolvDataset("test")
    test_loader = ter.DataLoader(test_dataset, batch_size=10, shuffle=True)

    # Initializing the modified TensorNet model, loss function, and optimizer
    model = TensorNet(
        hidden_channels=128,
        num_layers=2,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation='silu',
        equivariance_invariance_group='O(3)',
        cutoff_lower=0.0, # Default 
        cutoff_upper=4.5,  # Default  
        max_num_neighbors= 100 #Default is 64    
    )
    criterion = nn.MSELoss()  # Example loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    

    # Using early_stopper class
    early_stopper = EarlyStopper(patience=5, min_delta=0.001)
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(3):
        model.train()
        print(f"Current Epoch: {epoch}")
        running_loss = 0.0
        for batch in tqdm(train_loader):
            true_forces = batch.forces.to(device) ## True value of forces
            optimizer.zero_grad()
            try:
                # Forward pass
                force = model(batch.atomic_numbers.to(device), batch.positions.to(device), batch.batch.to(device), box=None)
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                #print(f"atomic_numbers type: {type(atomic_numbers)}, positions type: {type(positions)}, batch type: {type(batch_indices)}, box type: {type(box)}, charges type: {type(charges)}")
                raise
            forces_pred=force[3].requires_grad_()
            loss = criterion(forces_pred, true_forces)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Clear cache and garbage collect to free memory
            torch.cuda.empty_cache()
            gc.collect()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        print(f"Training Loss: {train_loss}")
        print(f"Training Loss is: {running_loss / len(train_loader)}")
        #Validation starts
        model.eval()
        val_loss=0
        with torch.no_grad():
            for batch in val_loader:
                true_forces = batch.forces.to(device)
                try:
                    force = model(batch.atomic_numbers.to(device), batch.positions.to(device), batch.batch.to(device), box=None)
                except Exception as e:
                    print(f"Error during model forward pass: {e}")
                    #print(f"atomic_numbers type: {type(atomic_numbers)}, positions type: {type(positions)}, batch type: {type(batch_indices)}, box type: {type(box)}, charges type: {type(charges)}")
                    raise
                forces_pred=force[3].requires_grad_()
                #print(f"Shape of forces_pred: {forces_pred.shape}")
                #print(f"Shape of true_forces: {true_forces.shape}")

                loss = criterion(forces_pred, true_forces)
                val_loss += loss.item()
                torch.cuda.empty_cache()
                gc.collect()
            print(f"Validation Loss is: {val_loss / len(val_loader)}")
            
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss}")

        # Early stopping check
        if early_stopper.early_stop(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    plot_loss(train_losses, val_losses, epoch)
    print("Validation is complete.")
    torch.save(model.state_dict(),"validated_model.pt")
    torch.cuda.empty_cache() 

# Plotting the graph for Train_loss vs Epoch and Val_loss vs Epoch
# Also, plotting early_stopping epoch criterion.
def plot_loss(train_losses, val_losses, early_stopping_epoch):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.axvline(early_stopping_epoch, color='g', linestyle='--', label='Early Stopping')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    train()

