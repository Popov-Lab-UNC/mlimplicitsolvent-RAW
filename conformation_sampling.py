from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import mdtraj as md
import alchemlyb
from alchemlyb.estimators import MBAR
import numpy as np
from openmm.unit import *
import pickle as pkl
from openff.toolkit.topology import Molecule,Topology
from scipy.optimize import curve_fit
import os
import torch as pt
import torch
from openmm import app, NonbondedForce, LangevinMiddleIntegrator, Platform, LangevinIntegrator
from openmm.app import Simulation, PDBFile, Topology
from freesolv_helper import FreesolvHelper
from MachineLearning.GNN_Models import GNN3_scale_96
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from tqdm import tqdm
from openmmtools.constants import kB
import alchemlyb.preprocessing
import time

class conformational_sampling:

    @staticmethod
    def sigmoid(x, L, x0, k, b):
        y = L / (1 + np.exp(-k*(x - x0))) + b
        return y

    def __init__(self, path, smile, name, model, dcd_size):
        self.dcd_size = dcd_size
        self.lambda_electrostatics = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8,
            0.85, 0.9, 1]
        self.lambda_sterics = [
            0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8,
            0.85, 0.9, 1
        ]
        self.smile = smile
        self._T = 300 * kelvin
        self.name = name

        full_path = os.path.join(path, name)

        if(not os.path.exists(full_path)):
            raise Exception("Path does not Exist")
        
        self.conform_path = os.path.join(full_path, "conformational_collection")
        os.makedirs(self.conform_path, exist_ok=True)


        self.solv_path = os.path.join(full_path, name + "_solv")

        model_path = os.path.join(self.solv_path, name+"_gnn_paramed_model.pt")
        params_path = os.path.join(self.solv_path, name+"_gnn_params.pkl")



        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if (self.device == torch.device("cpu")):
            self.platform = Platform.getPlatformByName('CPU')
        else:
            self.platform = Platform.getPlatformByName('CUDA')

        tot_unique = [
            0.14, 0.117, 0.155, 0.15, 0.21, 0.185, 0.18, 0.17, 0.12, 0.13
        ]
        model_dict = pt.load(model, map_location=self.device)
        self.model = GNN3_scale_96(max_num_neighbors=10000,
                                   parameters=None,
                                   device=self.device,
                                   fraction=0.5,
                                   unique_radii=tot_unique,
                                   jittable=True).to(self.device)
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)
        self.model.eval()

        with open(params_path, "rb") as f:
            self.gnn_params = pkl.load(f)

    def get_solv_lambda_schedule(self):
        """ Returns a list of tuples of (lambda_ster, lambda_elec) 
        for the solvation simulations """

        lambda_schedule = []
        lambda_ster = 1.0
        for lambda_elec in reversed(self.lambda_electrostatics):
            lambda_schedule.append((lambda_ster, lambda_elec))

        lambda_elec = 0.0
        for lambda_ster in reversed(self.lambda_sterics):
            lambda_schedule.append((lambda_ster, lambda_elec))

        return lambda_schedule


    def run_u_nk(self, traj, e_lambda_ster, e_lambda_elec):
        e_lambda_ster = torch.scalar_tensor(e_lambda_ster).to(self.device)
        e_lambda_elec = torch.scalar_tensor(e_lambda_elec).to(self.device)
        u = np.zeros(len(traj.time))
        for idx, coords in enumerate(traj.xyz):
            positions = torch.from_numpy(coords).to(self.device)
            batch = torch.zeros(size=(len(positions), )).to(torch.long) 

            factor = self.model(positions, e_lambda_ster, e_lambda_elec,
                                torch.tensor(1.0).to(self.device), True, batch, self.gnn_params)
            self.curr_simulation_vac.context.setPositions(coords)
            self.curr_simulation_vac.minimizeEnergy()
            U = self.curr_simulation_vac.context.getState(
                getEnergy=True).getPotentialEnergy()
            val = (U +
                   (factor[0].item() * kilojoule_per_mole)) / (kB * self._T)            

            u[idx] = float(val)
        return u
       

    def u_nk_processing_df(self, df):
        df.attrs = {
            "temperature": self._T,
            "energy_unit": "kT",
        }

        df = alchemlyb.preprocessing.decorrelate_u_nk(df, remove_burnin=True)
        return df


    def create_system(self, pdb_path):
        molecule = Molecule.from_smiles(self.smile)
        pdb = app.PDBFile(pdb_path)
        smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule)
        forcefield = app.ForceField()
        forcefield.registerTemplateGenerator(smirnoff.generator)
        system_F = forcefield.createSystem(pdb.topology)
        return system_F, molecule, pdb.topology


    def solv_u_nk(self):

        solv_u_nk_dfs = [[] for _ in self.dcd_size]

        integrator = LangevinIntegrator(self._T, 1 / picosecond,
                                        0.001 * picoseconds)

        system, mol, topology = self.create_system(os.path.join(self.solv_path, name+".pdb"))

        self.curr_simulation_vac = Simulation(topology, system, integrator, self.platform)





        for (lambda_ster,
            lambda_elec) in tqdm(self.get_solv_lambda_schedule()):
            dcd_file = os.path.join(
                self.solv_path,
                f"({lambda_ster}-{lambda_elec})_{self.name}.dcd")
            pdb_file = os.path.join(self.solv_path, f"{self.name}.pdb")
            traj = md.load(dcd_file, top=pdb_file)

            #I made 2500 steps by accident, whoops
            traj = traj[:3] #removed for 1000 check

            all_u_values = {}

            for (e_lambda_ster,
                 e_lambda_elec) in self.get_solv_lambda_schedule():
                u = self.run_u_nk(traj, e_lambda_ster,
                                                   e_lambda_elec)
                all_u_values[(e_lambda_ster, e_lambda_elec)] = u


            for idx, length in enumerate(self.dcd_size):
                if length > len(traj.time):
                    continue
                df = pd.DataFrame({
                "time": traj.time[:length],
                "vdw-lambda": [lambda_ster] * length,
                "coul-lambda": [lambda_elec] * length,
                })
                df = df.set_index(["time", "vdw-lambda", "coul-lambda"])

                for (e_lambda_ster, e_lambda_elec), u_values in all_u_values.items():
                    df[(e_lambda_ster, e_lambda_elec)] = u_values[:length]

                df = self.u_nk_processing_df(df)
                solv_u_nk_dfs[idx].append(df)

        for idx, solv_u_nk_df in enumerate(solv_u_nk_dfs):
            solv_u_nk_df = alchemlyb.concat(solv_u_nk_df)

            new_index = []
            for i, index in enumerate(solv_u_nk_df.index):
                new_index.append((i, *index[1:]))
            solv_u_nk_df.index = pd.MultiIndex.from_tuples(
                new_index, names=solv_u_nk_df.index.names)

            solv_u_nk_df.to_pickle(os.path.join(self.conform_path, f"solv_u_nk_{self.dcd_size[idx]}.pkl"))

        return solv_u_nk_dfs



    def compute_delta_F(self):

        print(" -- Starting Calculation of Hydration Energy -- ")

        solv = self.solv_u_nk()
        calcs = []
        for calc in solv:

            mbar_solv = MBAR()
            mbar_solv.fit(calc[0])


            print(mbar_solv.overlap_matrix)

            F_solv_kt = mbar_solv.delta_f_[(0.0, 0.0)][(
            1.0, 1.0)]  #mbar_vac.delta_f_[0][1] -
            F_solv = F_solv_kt * self._T * kB

            calcs.append(-F_solv.value_in_unit(kilojoule_per_mole) * 0.239006)
        
        print(f"--All Calculations for {self.name}--")
        print(calcs)


    def compute_lambda_delta_F(self):

        solv = self.solv_u_nk()
        calcs = []
        lambda_ster = self.lambda_sterics
        lambda_elec = self.lambda_electrostatics

        for elec_branch in range(len(self.lambda_electrostatics)):
            self.lambda_electrostatics

        self.lambda_sterics = lambda_ster[:]




import sys

if __name__ == '__main__':


    smile = str(sys.argv[1])
    expt = float(sys.argv[2])
    name = str(sys.argv[3])
    start = time.time()
    dcd_size = [3]
    path = "/work/users/r/d/rdey/LSNN_250k"
    model_path = '/work/users/r/d/rdey/ml_implicit_solvent/trained_models/280KDATASET2Kv3model.dict'
    main = conformational_sampling(path, smile, name, model_path, dcd_size)
    calcs = main.compute_delta_F()
    print(f"Time: {time.time() - start}")

    

    


    


    

