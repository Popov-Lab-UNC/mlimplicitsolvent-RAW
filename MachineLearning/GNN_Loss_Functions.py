"""
File defining LosssFunctions for the training of GNNs
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from config import CONFIG


def calc_all_losses(pre_energy, pre_forces, pre_sterics, pre_electrostatics, ldata):
    loss_f = F.mse_loss(pre_forces, ldata.forces)
    loss_sterics = F.mse_loss(pre_sterics, ldata.sterics_derivative)
    loss_elec = F.mse_loss(pre_electrostatics, ldata.electrostatics_derivative)
    tot_loss = (
        loss_f * CONFIG.loss.force_weight
        + loss_sterics * CONFIG.loss.sterics_weight
        + loss_elec * CONFIG.loss.electrostatics_weight
    )

    loss_dict = {
        "loss": tot_loss,
        "force_loss": loss_f,
        "sterics_loss": loss_sterics,
        "electrostatics_loss": loss_elec,
    }

    return tot_loss, loss_dict
