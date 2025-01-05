from collections import defaultdict
import time
import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd
from config import CONFIG
from omegaconf import OmegaConf
import torch
from torch_geometric.loader import DataLoader
# from terrace import DataLoader
from torch_geometric.data import Data
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import os, sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import wandb
import matplotlib.pyplot as plt

try:
    from GNN_Models import *
except:
    from MachineLearning.GNN_Models import *

import terrace as ter
from sklearn.metrics import mean_squared_error

from functools import lru_cache
from Simulation.Simulator import Simulator
from openmm.app.internal.customgbforces import GBSAGBn2Force
from ForceField.Forcefield import Vacuum_force_field, OpenFF_forcefield_GBNeck2
from openmm import NonbondedForce
from Data.Datahandler import hdf5_storage
from MachineLearning.GNN_Loss_Functions import *
from torchmetrics import R2Score

# import matplotlib inset

try:
    from GNN_Graph import get_Graph_for_one_frame
except:
    from MachineLearning.GNN_Graph import get_Graph_for_one_frame


class Trainer:

    def __init__(
        self,
        name="name",
        path=".",
        verbose=False,
        enable_tmp_dir=True,
        force_mode=False,
        device=None,
        random_state=161311,
        use_wandb=True,
    ):
        self._name = name
        self._model = None
        self._optimizer = None
        if device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device
        self._path = path
        self._verbose = verbose
        self._random_state = random_state
        self._force_mode = force_mode
        self._use_wandb = use_wandb

        if enable_tmp_dir == True:
            try:
                self._tmp_folder = os.environ["TMPDIR"] + "/"
            except:
                enable_tmp_dir = False
        self._use_tmpdir = enable_tmp_dir
        self._tmpdir_in_use = False
        self._explicit_data = False
        self.check = []
        self._model_path = self._path + "/" + self._name + ".pt"

    def set_lossfunction(self, lossfunction=None):

        if lossfunction is None:
            self.calculate_loss = self.calculate_loss_default
        else:
            self.calculate_loss = lossfunction

    def calculate_loss_default(self, pre_energy, pre_forces, ldata):

        # For summed energies compare to sum of energies
        if pre_energy.size() == torch.Size([]):
            energy_loss = F.mse_loss(pre_energy, ldata.energies.sum())
        else:
            energy_loss = F.mse_loss(pre_energy.unsqueeze(1), ldata.energies)
        force_loss = F.mse_loss(pre_forces, ldata.forces)
        el_val = energy_loss.tolist()
        fl_val = force_loss.tolist()

        loss_e = (fl_val / (el_val + fl_val)) * energy_loss * 1 / 4
        loss_f = (el_val / (el_val + fl_val)) * force_loss * 3 / 4

        loss = loss_e + loss_f
        # print(el_val,fl_val)

        return loss

    def init_wandb(self, name):

        if not self._use_wandb:
            return

        run = wandb.init(
            project="ML Implicit Solvent",
            entity="molecularmodellinglab",
            name=name,
            config=OmegaConf.to_object(CONFIG),
        )

    def log(self, split, d):
        d = {f"{split}_{key}": value for key, value in d.items()}
        if self._use_wandb:
            wandb.log(d)

    def create_metrics(self):
        """ Create TorchMetrics for a single train or val epoch """
        return nn.ModuleDict({
            "r2_forces": R2Score(),
            "r2_sterics": R2Score(),
            "r2_electrostatics": R2Score(),
            # metrics when lambda_sterics and lambda_electrostatics are 1
            "r2_forces_lambda_1": R2Score(),
            "r2_sterics_lambda_1": R2Score(),
            "r2_electrostatics_lambda_1": R2Score(),
        }).to(self._device)

    def update_metrics(self, metrics, pre_energy, pre_forces, pre_sterics,
                       pre_electrostatics, ldata, mask_sterics,
                       mask_electrostatics):
        """ Update the metrics for a single batch and return the metric values """

        lambda_1_mask = (ldata.lambda_sterics
                         == 1) & (ldata.lambda_electrostatics == 1)
        lambda_1_mask_exp = lambda_1_mask[ldata.batch]
        ret = {}
        ret["r2_forces"] = metrics["r2_forces"](pre_forces.flatten(),
                                                ldata.forces.flatten())
        ret["r2_sterics"] = metrics["r2_sterics"](
            pre_sterics[mask_sterics].flatten(),
            ldata.sterics_derivative[mask_sterics].flatten())
        ret["r2_electrostatics"] = metrics["r2_electrostatics"](
            pre_electrostatics[mask_electrostatics].flatten(),
            ldata.electrostatics_derivative[mask_electrostatics].flatten())
        if lambda_1_mask.sum() > 1:
            ret["r2_forces_lambda_1"] = metrics["r2_forces_lambda_1"](
                pre_forces[lambda_1_mask_exp].flatten(),
                ldata.forces[lambda_1_mask_exp].flatten())
            ret["r2_sterics_lambda_1"] = metrics["r2_sterics_lambda_1"](
                pre_sterics[lambda_1_mask].flatten(),
                ldata.sterics_derivative[lambda_1_mask].flatten())
            ret["r2_electrostatics_lambda_1"] = metrics[
                "r2_electrostatics_lambda_1"](
                    pre_electrostatics[lambda_1_mask].flatten(),
                    ldata.electrostatics_derivative[lambda_1_mask].flatten())
        else:
            print(
                "WARNING: lambda_1_mask.sum() <= 1. This may cause validation issues."
            )

        return ret

    def train_model(
        self,
        runs,
        batch_size=100,
        clip_gradients=0,
    ):
        #torch.autograd.set_detect_anomaly(True)
        assert self._optimizer != None
        assert self._model != None
        self.init_wandb(self._name)
        self._model.train()

        val_results = []

        loader = DataLoader(
            self._training_data,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
        )

        # loader.to(device)
        for i in range(runs):
            start = time.time()
            total_loss = []
            train_metrics = self.create_metrics()
            metric_buffer = defaultdict(list)

            pbar = tqdm(loader)
            for d, ldata in enumerate(pbar):
                # set optimizer gradients
                self._optimizer.zero_grad()
                ldata = ldata.to(self._device)
                # Make prediction
                pre_energy, pre_forces, pre_sterics, pre_electrostatics = self._model(
                    ldata.pos,
                    ldata.lambda_sterics,
                    ldata.lambda_electrostatics,
                    torch.tensor(0.0),
                    False,
                    ldata.batch,
                    ldata.atom_features,
                )

                mask_sterics = True  #(ldata.lambda_sterics != 0.0) & (ldata.lambda_sterics != 1.0)
                mask_electrostatics = True  #(ldata.lambda_electrostatics != 0.0) & (ldata.lambda_electrostatics != 1.0)

                loss, metric_dict = self.calculate_loss(
                    pre_energy=pre_energy,
                    pre_forces=pre_forces,
                    pre_sterics=pre_sterics,
                    pre_electrostatics=pre_electrostatics,
                    ldata=ldata,
                    mask_sterics=mask_sterics,
                    mask_electrostatics=mask_electrostatics)
                '''
                if metric_dict["sterics_loss"] > 20000:
                    print(ldata.lambda_sterics[mask_sterics], 
                        ldata.sterics_derivative[mask_sterics], 
                        pre_sterics[mask_sterics], 
                        metric_dict["sterics_loss"],
                        metric_dict["force_loss"],
                        )
                '''
                total_loss.append(loss.item())
                assert torch.isnan(loss).sum() == 0
                loss.backward()

                metric_dict.update(
                    self.update_metrics(
                        train_metrics,
                        pre_energy,
                        pre_forces,
                        pre_sterics,
                        pre_electrostatics,
                        ldata,
                        mask_sterics,
                        mask_electrostatics,
                    ))

                for key, value in metric_dict.items():
                    metric_buffer[key].append(value.item())

                if d % CONFIG.logging_freq == 0:
                    self.log(
                        "train", {
                            key: np.mean(value)
                            for key, value in metric_buffer.items()
                        })
                    metric_buffer = defaultdict(list)

                if clip_gradients != 0:
                    clip_grad_norm_(self._model.parameters(), clip_gradients)

                self._optimizer.step()
                pbar.set_description("Run %i avg time: %f3 loss: %f3" % (
                    i,
                    (time.time() - start) / (d + 1),
                    np.nanmean(total_loss),
                ))
                #break
            #continue

            val_results.append(self.validate_model(batch_size=batch_size))
            print(f"Validation results for epoch {i}")
            for key, value in val_results[-1].items():
                print(f"   {key}: {value:.3f}")
            self.log("val", val_results[-1])
            self._scheduler.step(val_results[-1]["loss"])

        # self.save_training_log([], Val_Losses)

        return [], val_results

    def identify_problematic_sets(self, error_tolerance=5000):

        list_of_problematic_entries = []

        for i, data in enumerate(self._validation_data):
            loader = DataLoader(data, batch_size=1)

            for l, ldata in enumerate(loader):
                ldata.to(self._device)
                pre_energy, pre_forces = self._model(ldata)
                loss = self.calculate_loss(pre_energy, pre_forces, pre_sterics,
                                           pre_electrostatics, ldata)
                if np.max(loss.to("cpu").tolist()) > error_tolerance:
                    list_of_problematic_entries.append(
                        [ldata.smiles, ldata.molid, ldata.confid, ldata.hdf5])

        for i, data in enumerate(self._training_data):
            loader = DataLoader(data, batch_size=1)

            for l, ldata in enumerate(loader):
                ldata.to(self._device)
                pre_energy, pre_forces = self._model(ldata)
                loss = self.calculate_loss(pre_energy, pre_forces, ldata)
                if np.max(loss.to("cpu").tolist()) > error_tolerance:
                    list_of_problematic_entries.append(
                        [ldata.smiles, ldata.molid, ldata.confid, ldata.hdf5])

        return list_of_problematic_entries

    def validate_model(self, batch_size):
        metrics = self.create_metrics()
        loader = DataLoader(self._validation_data, batch_size=batch_size)
        losses = defaultdict(list)
        derivatives = []
        lambdas = []
        for l, ldata in enumerate(tqdm(loader)):
            ldata.to(self._device)
            pre_energy, pre_forces, pre_sterics, pre_electrostatics = self._model(
                ldata.pos,
                ldata.lambda_sterics,
                ldata.lambda_electrostatics,
                torch.tensor(0.0),
                False,
                ldata.batch,
                ldata.atom_features,
            )
            mask_sterics = (ldata.lambda_sterics
                            != 0.0) & (ldata.lambda_sterics != 1.0)
            mask_electrostatics = (ldata.lambda_electrostatics != 0.0) & (
                ldata.lambda_electrostatics != 1.0)
            loss, loss_dict = self.calculate_loss(pre_energy, pre_forces,
                                                  pre_sterics,
                                                  pre_electrostatics, ldata,
                                                  mask_sterics,
                                                  mask_electrostatics)

            lambdas.append(ldata.lambda_electrostatics.detach().to('cpu'))
            derivatives.append(pre_electrostatics.detach().to('cpu'))
            for key, value in loss_dict.items():
                losses[key].append(value.item())

            self.update_metrics(metrics, pre_energy, pre_forces, pre_sterics,
                                pre_electrostatics, ldata, mask_sterics,
                                mask_electrostatics)

        results = {key: np.mean(value) for key, value in losses.items()}
        '''
        results.update(
            {
                key: metrics[key].compute()
                for key in metrics.keys()
            }
        )
        '''
        return results

    def save_model(self):

        torch.save(self._model, self._path + "/" + self._name + ".pt")

    @property
    def model_path(self):

        return self._path + "/" + self._name + ".pt"

    def save_dict(self):
        self._model.eval()
        torch.save(self._model.state_dict(),
                   self._path + "/" + self._name + "model.dict")

    def load_dict(self):
        self._model.load_state_dict(
            torch.load(self._path + "/" + self._name + "model.dict"))
        self._model.eval()

    def load_model(self, path=None):
        if path is None:
            assert os.path.isfile(self._path + "/" + self._name + ".pt")
            self._model = torch.load(self._path + "/" + self._name + ".pt")
        else:
            self._model = torch.load(path)
        # self._model.eval()

        

    def initialize_optimizer(self, lr, schedule=None):
        assert self._model != None

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and torch.cuda.is_available()
        extra_args = dict(fused=True) if use_fused else dict()
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=lr, **extra_args
        )

        if schedule == "Plateau":
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizer, verbose=True, factor=0.8
            )
        elif schedule == "Exponential":
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self._optimizer, gamma=0.01 ** (1 / 1000)
            )
        elif schedule == "Exponential100":
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self._optimizer, gamma=0.01 ** (1 / 100)
            )
        elif schedule == "Exponential10":
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self._optimizer, gamma=0.01 ** (1 / 10)
            )
        elif schedule == "Exponential30":
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self._optimizer, gamma=0.01 ** (1 / 30)
            )
        elif schedule == "Exponential50":
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self._optimizer, gamma=0.01 ** (1 / 50)
            )
        else:
            self._scheduler = Dummy_scheduler()
