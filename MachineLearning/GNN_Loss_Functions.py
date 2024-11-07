"""
File defining LosssFunctions for the training of GNNs
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from config import CONFIG


def calc_all_losses(pre_energy, pre_forces, pre_sterics, pre_electrostatics, ldata, mask_sterics, mask_electrostatics):
    #cheap workaround to dataset issues
    mask_nan = ~torch.isnan(pre_forces) & ~torch.isnan(ldata.forces)

    loss_f = F.mse_loss(pre_forces[mask_nan], ldata.forces[mask_nan])
    loss_sterics = F.mse_loss(pre_sterics.view(-1,)[mask_sterics], ldata.sterics_derivative[mask_sterics])
    loss_elec = F.mse_loss(pre_electrostatics.view(-1,)[mask_electrostatics], ldata.electrostatics_derivative[mask_electrostatics])
    #print(pre_sterics[mask_sterics], ldata.sterics_derivative[mask_sterics], loss_sterics.item())
    #print(pre_electrostatics[mask_electrostatics], ldata.electrostatics_derivative[mask_sterics], loss_elec.item())

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

    if torch.isnan(tot_loss).sum() != 0:
        torch.set_printoptions(profile="full")
        if torch.isnan(loss_f).sum() != 0:
            print("nan found in Forces")
            print("Forces")
            print(pre_forces, ldata.forces)
            print("Energy")
            print(pre_energy)
            print("Positions")
            print(ldata.pos)

            for idx, coord in enumerate(pre_forces):
                if torch.isnan(coord).sum() != 0:
                    print("flawed data")
                    print(ldata.forces[idx])
                    print(coord)
                    print(ldata.pos[idx])


        
    

    return tot_loss, loss_dict
