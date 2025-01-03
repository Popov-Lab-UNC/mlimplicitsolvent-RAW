import pandas as pd
import yaml
import os
import shutil
from simple_slurm import Slurm
import datetime
import subprocess
from rdkit import Chem
import numpy as np


class FreesolvHelper:

    def __init__(self, file_path):
        self._fp = file_path

    def csv_reader(self):
        df = pd.read_csv(self._fp)
        return df

    def run_simulation(smiles):
        return None

    def solvation_calculation(smiles):
        return None

    def smiles_reader(self, save=False):
        df = self.csv_reader()
        res = []
        for index, row in df.iterrows():
            smiles_string = row['smiles']
            expt_string = row['expt']
            name_string = row['iupac']
            res.append([smiles_string, expt_string, name_string])
        return res

    def smiles_file_creation(self, file_name, smiles_string):
        print(smiles_string)
        with open(file_name, 'w') as file:
            file.write(smiles_string + '\n')

    def smiles_conversion(self):
        folder = "/work/users/r/d/rdey/mol2_files"
        smiles = []
        for idx, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                print(f"Failed on {smile}")
                continue
            mol = Chem.AddHs(mol)
            file_path = os.path.join(folder, f"{idx}.mol2")
            Chem.MolToMolFile(mol, file_path)
