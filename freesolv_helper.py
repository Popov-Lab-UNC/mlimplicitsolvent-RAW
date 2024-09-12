
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

    def smiles_reader(self, save = False):
        file_name = "/nas/longleaf/home/rdey/ml_implicit_solvent/yank_smiles.smiles"
        # Iterating through rows using iterrows()
        df = self.csv_reader()
        res = []
        for index, row in df.iterrows():
            smiles_string = row['smiles']
            expt_string = row['expt']
            name_string = row['iupac']
            res.append([smiles_string, expt_string, name_string])
            if save:
                self.smiles_file_creation(file_name, smiles_string)
        return res

    def smiles_file_creation(self, file_name, smiles_string):
        print(smiles_string)
        with open(file_name, 'w') as file:
            file.write(smiles_string +'\n')

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

            
    def mol2Total(self):
        folder_name = "/work/users/r/d/rdey/ml_implicit_solvent/yank_simulations/mol2files/mol2files_gaff"
        mol2_rel_path = [file for file in os.listdir(folder_name)]
        mol2_paths = [os.path.join(folder_name, file) for file in os.listdir(folder_name)]
        mol2_name = []
        for mol2_path in mol2_paths:
            with open(mol2_path, 'r') as file: 
                file.readline()
                mol2_name.append(file.readline().strip())

        return mol2_paths, mol2_rel_path, mol2_name



    def yank_script_generation(self):
        mol2_paths, mol2_rel_path, mol2_name = self.mol2Total()
        folder = "/work/users/r/d/rdey/yank_files"
        original = os.path.join(folder, "yank_solv.yaml")
        for idx, mol2 in enumerate(mol2_paths):
            copy = os.path.join(folder, f"{idx}.yaml")
            shutil.copyfile(original, copy)
            with open(copy, 'r') as file:
                config = yaml.safe_load(file)

            config['molecules']['name']['filepath'] = mol2
            config['options']['output_dir'] = f"{idx}_yaml"

            with open(copy, 'w') as yaml_file:
                yaml.dump(config, yaml_file)



    def slurm_generation(self):
        folder = "/work/users/r/d/rdey/yank_files"
        for i in range(642):
            slurm = Slurm(
                    partition = "l40-gpu",
                    nodes=1,
                    ntasks=1,
                    job_name=f'yank_solv{i}',
                    time=datetime.timedelta(days=0, hours=1, minutes=0, seconds=0),
                    mem = "1g",
                    qos = "gpu_access",
                    gres = "gpu:1"
                )
            file = os.path.join(folder, f"{i}.yaml")
            #file = "implicit/experiments"
            #command = f'yank analyze --store={file}'
            command = f'yank script --yaml={file}'
            slurm.add_cmd(f'source ~/.bashrc && mamba activate yank && {command}')
            slurm.sbatch()


    def yankl_analysis_output_generation(self):
        mol2_paths, mol2_rel_path, mol2_name = self.mol2Total()
        folder = "/work/users/r/d/rdey/yank_files/"
        idxs = []
        for idx, mol2 in enumerate(mol2_rel_path):
            file_path = os.path.join(folder, f"{idx}_yaml/experiments")
            output = os.path.join(f"{folder}00_output_files/",f"{idx}_output.txt")
            if(not os.path.exists(output)):
                print(idx)
                command = f"yank analyze --store={file_path}"
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    idxs.append(idx)
                    with open(output, "w") as file:
                        file.write(result.stdout)
            else:
                idxs.append(idx)
        return(idxs)




    def read_analysis(self):
        folder = "/work/users/r/d/rdey/yank_files/00_output_files/"
        energies = []
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            with open(file_path, 'r') as file:
                line = file.readline()

                if "Free energy of solvation:" in line: 
                    start = (line.find('(') + 1)
                    end = (line.rfind("+"))
                    value = line[start: end].split()[0]
                    energies.append(float(value))
        return energies


    #print(read_analysis())

    #_,_, mol2_name = mol2Total()

    #print(mol2_name)

    #yank_analysis_output_generation()



