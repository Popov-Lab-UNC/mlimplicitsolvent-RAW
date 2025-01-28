import torch
from torch_geometric.loader import DataLoader
from datasets.bigbind_solv import MAFBigBind
import terrace as ter
import sys

from MachineLearning.GNN_Trainer import Trainer
from MachineLearning.GNN_Models import *
from MachineLearning.GNN_Loss_Functions import *
from config import CONFIG, load_config

if len(sys.argv) > 1:
    load_config(sys.argv[1])

print(f"Name of Model: {CONFIG.name}")

trainer = Trainer(
    verbose=True,
    name=CONFIG.name,
    path="trained_models",
    force_mode=True,
    enable_tmp_dir=False,
    use_wandb=CONFIG.use_wandb,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tot_unique = [0.14, 0.117, 0.155, 0.15, 0.21, 0.185, 0.18, 0.17, 0.12, 0.13]
model = GNN3_scale_96(max_num_neighbors=10000,
                      parameters=None,
                      device=device,
                      fraction=0.5,
                      unique_radii=tot_unique,
                      jittable=True)
trainer._training_data = MAFBigBind(
    "train_no_charges", dir='/work/users/r/d/rdey/BigBind500k/bigbind_solv/')
trainer._validation_data = MAFBigBind(
    "test_re_no_charges", dir='/work/users/r/d/rdey/BigBind500k/bigbind_solv/')
trainer._model = model

trainer.initialize_optimizer(CONFIG.learn_rate, CONFIG.lr_scheduler)
trainer.set_lossfunction(calc_all_losses)

trainer.train_model(
    runs=CONFIG.num_epochs,
    batch_size=CONFIG.batch_size,
    clip_gradients=CONFIG.clip_gradients,
)

trainer.save_model()
trainer.save_dict()
