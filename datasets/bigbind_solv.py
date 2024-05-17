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
        positions = group["positions"][:]
        atomic_numbers = group["atomic_numbers"][:]
        forces = group["solv_forces"][:]
        return q, positions, atomic_numbers, forces