import os
import alchemlyb.preprocessing
from openff.toolkit.topology import Molecule, Topology
from openmmforcefields.generators import GAFFTemplateGenerator
from openmmtorch import TorchForce
from openmm.app.internal.customgbforces import GBSAGBn2Force
import torch
from openmm import app
from openmm.app import *
import alchemlyb
from alchemlyb.estimators import MBAR
import numpy as np
from openmm.unit import *
from openmm import LangevinMiddleIntegrator
from openmm.app.dcdreporter import DCDReporter
from openmmtools.constants import kB
import pandas as pd
from freesolv_helper import FreesolvHelper
from MachineLearning.GNN_Models import *
import shutil
import mdtraj as md 

class AI_Solvation_calc:

    @staticmethod
    #taken from Michael
    def get_ligand_and_water_indices(system):
        lig_indices = []
        water_indices = []
        for residue in system.topology.residues():
            in_lig = False
            atom_indices = set()
            for atom in residue.atoms():
                if atom.index in system.lig_indices:
                    in_lig = True
                atom_indices.add(atom.index)
            if in_lig:
                lig_indices.append(atom_indices)
            else:
                water_indices.append(atom_indices)

        return lig_indices, water_indices
    
    @staticmethod
    def create_system_from_smiles(smiles, lambda_electrostatics):
        molecule = Molecule.from_smiles(smiles)
        for atom, p_charge in zip(molecule.atoms, molecule.partial_charges):
            atom.partial_charge = p_charge*lambda_electrostatics
        gaff = GAFFTemplateGenerator(molecule=molecule)
        forcefield = app.Forcefield()
        forcefield.registerTemplateGenerator(gaff.generator)
        system = forcefield.createSystem(molecule.topology, nonBondedMethod = app.NoCutoff)
        return system, Molecule

    

    def __init__(self, model, smiles, path):
        self.lambda_sterics = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.lambda_electrostatics = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.n_steps = 10000
        self.report_interval = 1000
        self._T = 300*kelvin
        self.model = torch.jit.load(model)
        self.model.eval()
        self.smiles = smiles
        
        self.path = path
        self.solv_path = os.path.join(self.path, f"{self.smiles}_solv")
        self.vac_path = os.path.join(self.path, f"{self.smiles}_vac")

    def set_model(self):
        assert self.system
        self.model.set_gnn_params(self.compute_atom_features)
        self.model_force = TorchForce(self.model)
        self.model_force.addGlobalParameter("lambda_sterics", 1.0)
        self.model_force.addGlobalParameter("lambda_electrostatics", 1.0)
        self.model_force.addGlobalParameter("vaccum", 0.0)
        self.model_force.addGlobalParameter('testing', 0.0)
        self.model_force.setOutputsForces(True)


    #taken from Michael
    def get_solv_lambda_schedule(self):
        """ Returns a list of tuples of (lambda_ster, lambda_elec) 
        for the solvation simulations """

        lambda_schedule = []
        lambda_ster = 1.0
        for lambda_elec in reversed(self.electrostatics_schedule):
            lambda_schedule.append((lambda_ster, lambda_elec))
        
        lambda_elec = 0.0
        for lambda_ster in reversed(self.sterics_schedule):
            lambda_schedule.append((lambda_ster, lambda_elec))

        return lambda_schedule
        

    def set_system(self, lambda_electrostatics):
        self.system, self.molecule = self.create_system_from_smiles(self.smiles, lambda_electrostatics=lambda_electrostatics)
        self.topology = self.molecule.to_topology()

    def savePDB(self, path):
        assert self.molecule
        self.molecule.to_file(path, file_format= "pdb")

    def compute_atom_features(self):
        charges = np.array([tuple(f.getParticleParameters)[0] for f in self.system.getForces() if isinstance(f, app.NonbondedForce)][0])
        charges = np.array([self.system.getForces()[0].getParticleParameters(i)[0]._value for i in range(self.topology._numAtoms)])

        force = GBSAGBn2Force(cutoff=None,SA="ACE",soluteDielectric=1,solventDielectric=4)
        gbn2_parameters = np.empty((self.topology.getNumAtoms(),6))
        gbn2_parameters[:,0] = charges
        gbn2_parameters[:,1:] = force.getStandardParameters(self.topology)
        return gbn2_parameters
    
    def AI_simulation(self, lambda_sterics, lambda_electrostatics, vaccum, out):
        assert self.vac_path
        assert self.solv_path

        path = os.path.join(self.vac_path, f"{out}.dcd")
        com = os.path.join(self.solv_path, f"{out}.com")

        if os.path.exists(com):
            return 
        
        self.set_system(lambda_electrostatics)
        self.system.addForce(self.model_force)
        integrator = LangevinMiddleIntegrator(self._T, 1/unit.picosecond, 0.004*unit.picoseconds)
        simulation = Simulation(self.topology, self.system, integrator)
        simulation.context.setParameter("vaccum", vaccum)
        simulation.context.setParameter("lambda_electrostatics", lambda_electrostatics)
        simulation.context.setParameter("lambda_sterics", lambda_sterics)
        simulation.minimizeEnergy()
        simulation.reporters.append(DCDReporter(path, 100))
        simulation.step(self.n_steps)
        with open(com, 'w'):
            pass


    
    def run_all_sims(self, overwrite = False):
        self.set_system(1.0)
        self.set_model()
        self.AI_simulation(1.0, 1.0)
        
        
        if overwrite:
            shutil.rmtree(self.solv_path)
            shutil.rmtree(self.vac_path)

        if not os.path.exists(self.solv_path):
            os.mkdir(self.solv_path)
        if not os.path.exists(self.vac_path):
            os.mkdir(self.vac_path)

        pdb_path = os.path.join(self.solv_path, f"{self.smiles}.pdb")
        if not os.path.exists(pdb_path):
            self.savePDB(pdb_path) #ensure current system does not have changed electrostatics

        print("Removing electrostatics")
        for lambda_elec in reversed(self.lambda_electrostatics):
            self.AI_simulation(1.0, lambda_elec, vaccum = False, out = f"(1.0-{lambda_elec})_{self.smiles}")
        print("Removing sterics")
        for lambda_ster in reversed(self.lambda_sterics):
            self.AI_simulation(lambda_ster, 0.0, vaccum = False, out = f"({lambda_ster}-0.0)_{self.smiles}")
        print("Re-adding electrostatics")
        for lambda_elec in self.lambda_electrostatics:
            self.AI_simulation(0.0, lambda_elec, vaccum = True, out = f"{lambda_elec}_{self.smiles}")

    def calculate_energy_for_traj(self, traj, e_lambda_ster, e_lambda_elec):
        u = np.zeros(len(traj.time))
        for idx, coords in enumerate(traj.xyz):
            pre_energy, _, _, _ = self.model(coords, e_lambda_ster, e_lambda_elec, None, None, 1.0, 0.0)
            u[idx] = pre_energy/(kB*self._T)
        return u
    
    def u_nk_processing_df(self, df): 
        df.attrs = {
                "temperature": T.value_in_unit(unit.kelvin),
                "energy_unit": "kT",
            }
        df = alchemlyb.preprocessing.decorrelate_u_nk(df, remove_burnin=True)
        return df

    def solv_u_nk(self):
        cache_path = os.path.join(self.solv_path, f"{self.smiles}_u_nk.pkl")
        if os.path.exists(cache_path):
            return pd.read_pickle(cache_path)
        
        solv_u_nk_df = []
        for (lambda_ster, lambda_elec) in self.get_solv_lambda_schedule:
            dcd_file = os.path.join(self.solv_path, f"({lambda_ster}-{lambda_elec})_{self.smiles}.dcd")
            pdb_file  = os.path.join(self.solv_path, f"{self.smiles}.pdb")
            traj = md.load(dcd_file, top = pdb_file)

            df = pd.DataFrame({
                "time": traj.time,
                "vdw_lambda": [lambda_ster]*len(traj.time),
                "coul_lambda:": [lambda_elec]*len(traj.time),
            }
            )
            df = df.set_index(["time", "vdw-lambda", "coul-lambda"])

            for (e_lambda_ster, e_lambda_elec) in self.get_solv_lambda_schedule:
                u = self.calculate_energy_for_traj(traj, e_lambda_ster, e_lambda_elec)
                df[(e_lambda_ster, e_lambda_elec)] = u

            df = self.u_nk_processing_df(df)
            solv_u_nk_df.append(df)

        solv_u_nk_df = alchemlyb.concat(solv_u_nk_df)

        #not sure what this does, is this just to add an extra index into the thing??
        new_index = []
        for i, index in enumerate(solv_u_nk_df.index):
            new_index.append((i, *index[1:]))
        solv_u_nk_df.index = pd.MultiIndex.from_tuples(new_index, names=solv_u_nk_df.index.names)

        solv_u_nk_df.to_pickle(cache_path)

        return solv_u_nk_df
     
    def vac_u_nk(self):
        cache_path = os.path.join(self.vac_path, f"{self.smiles}_u_nk.pkl")
        if os.path.exists(cache_path):
            return pd.read_pickle(cache_path)
        vac_u_nk_df = []

        for lambda_elec in self.lambda_electrostatics:
            dcd_file = os.path.join(self.vac_path, f"{lambda_elec}_{self.smiles}.dcd")
            pdb_file =  os.path.join(self.solv_path, f"{self.smiles}.pdb")
            traj = md.load(dcd_file, top = pdb_file)

            df =  pd.DataFrame({
               "time": traj.time,
               "fep_lambda": [lambda_elec]*len(traj.time)
            })

            df.set_index(["time", "fep_lambda"])
            for (e_lambda_ster, e_lambda_elec) in self.get_solv_lambda_schedule:
                u = self.calculate_energy_for_traj(traj, e_lambda_ster, e_lambda_elec)
                df[(e_lambda_ster, e_lambda_elec)] = u
            df = self.u_nk_processing_df(df)
            vac_u_nk_df.append(df)
        vac_u_nk_df = alchemlyb.concat(vac_u_nk_df)

        new_index = []
        for i, index in enumerate(vac_u_nk_df.index):
            new_index.append((i, *index[1:]))
        vac_u_nk_df.index = pd.MultiIndex.from_tuples(new_index, names=vac_u_nk_df.index.names)
        vac_u_nk_df.to_pickle(cache_path)

        return vac_u_nk_df

    def compute_delta_F(self):
        solv = self.solv_u_nk
        vac = self.vac_u_nk
        mbar_vac = MBAR()
        mbar_vac.fit(vac)

        mbar_solv = MBAR()
        mbar_solv.fit(solv) 

        F_solv_kt = mbar_vac.delta_f_[0][1] - mbar_solv.delta_f_[(0,0)][(1,1)]
        F_solv = F_solv_kt*self._T*kB

        return F_solv

        
        
class runSims:

    def __init__(self, model, path):
        self.model = model
        self.collect = []
        self.path = path
    
    def run_all_smiles(self, smiles):
        for smile in smiles:
            obj = AI_Solvation_calc(self.model, smiles, path = self.path)
            obj.run_all_sims()
            res = obj.compute_delta_F()
            print(res)
            self.collect.append(res)
            del obj


    


if __name__ == "__main__":
    freesolv_calc = runSims('/work/users/r/d/rdey/ml_implicit_solvent/trained_models/trial1.model', "/work/users/r/d/rdey/trials")
    df = FreesolvHelper("/work/users/r/d/rdey/ml_implicit_solvent/freesolv/SAMPL.csv")
    df = df.smiles_reader()
    df = df[0]
    freesolv_calc.run_all_smiles(df)
 
