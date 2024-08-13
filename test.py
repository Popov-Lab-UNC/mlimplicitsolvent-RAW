import torch
from torch_geometric.loader import DataLoader
from datasets.bigbind_solv import BigBindSolvDataset
import terrace as ter

from MachineLearning.GNN_Trainer import Trainer
from MachineLearning.GNN_Models import *
from MachineLearning.GNN_Loss_Functions import *


trainer = Trainer(verbose=False,name='GNN3_pub_' + "Test1",path='trained_models',force_mode=True,enable_tmp_dir=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tot_unique = [0.14,0.117,0.155,0.15,0.21,0.185,0.18,0.17,0.12,0.13]
model = GNN3_scale_64(max_num_neighbors=10000,parameters=None,device=device,fraction=0.5,unique_radii=tot_unique)
trainer._training_data = BigBindSolvDataset("train", frame_index=10)
trainer._validation_data = BigBindSolvDataset("val", frame_index=1)
trainer.model = model

trainer.initialize_optimizer(0.0001,'Exponential30')
trainer.set_lossfunction(calculate_force_loss_only)

trainer.train_model(10, 8 ,5)
'''
trainer.save_model()
'''



