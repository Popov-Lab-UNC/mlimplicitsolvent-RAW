from datasets.md_batch import MDData
import torch
from torch.utils.data import Dataset
import h5py
import os
from config import CONFIG

class BigBindSolvDataset(Dataset):
    """ This dataset returns the charges, positions, atomic numbers,
    and forces of a frame in the bigbind_solv dataset."""

    def __init__(self, split):
        """ Split is either 'train', 'val', or 'test'."""
        file_path = os.path.join(CONFIG.bigbind_solv_dir, split + ".h5")
        self.file = h5py.File(file_path, "r")
        self.keys = list(self.file.keys())
        self.length = len(self.keys)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):

        group = self.file[self.keys[index]]
        q = group["charges"][:]
        all_positions = group["positions"][:]
        atomic_numbers = group["atomic_numbers"][:]
        all_forces = group["solv_forces"][:]

        # choose a random frame from the simulation
        frame_idx = torch.randint(0, all_positions.shape[0], (1,)).item()
        positions = all_positions[frame_idx]
        forces = all_forces[frame_idx]

        lambda_sterics = group["lambda_sterics"][frame_idx]
        lambda_electrostatics = group["lambda_electrostatics"][frame_idx]
        sterics_derivative = group["sterics_derivatives"][frame_idx]
        electrostatics_derivative = group["electrostatics_derivatives"][frame_idx]

        return MDData(
            charges=torch.tensor(q, dtype=torch.float32),
            positions=torch.tensor(positions, dtype=torch.float32),
            atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
            forces=torch.tensor(forces, dtype=torch.float32),
            lambda_sterics=torch.tensor(lambda_sterics, dtype=torch.float32),
            lambda_electrostatics=torch.tensor(lambda_electrostatics, dtype=torch.float32),
            sterics_derivative=torch.tensor(sterics_derivative, dtype=torch.float32),
            electrostatics_derivative=torch.tensor(electrostatics_derivative, dtype=torch.float32)
        )
