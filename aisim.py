import os
import shutil
import alchemlyb.preprocessing
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmmtorch import TorchForce
from openmm.app.internal.customgbforces import GBSAGBn2Force
import torch
from openmm import app, NonbondedForce, LangevinMiddleIntegrator, Platform, LangevinIntegrator
from openmm.app import *
import alchemlyb
from alchemlyb.estimators import MBAR
import numpy as np
from openmm.unit import *
from openmmtools.constants import kB
import pandas as pd
from freesolv_helper import FreesolvHelper
from MachineLearning.GNN_Models import GNN3_scale_96
from tqdm import tqdm
import mdtraj as md
import pickle as pkl
from rdkit import Chem
from rdkit.Chem import AllChem
import time


class AI_Solvation_calc:

    @staticmethod
    def create_system_from_smiles(smiles, pdb_path):
        molecule = Molecule.from_smiles(smiles)
        pdb = app.PDBFile(pdb_path)
        smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule)
        forcefield = app.ForceField()
        forcefield.registerTemplateGenerator(smirnoff.generator)
        system_F = forcefield.createSystem(pdb.topology)
        return system_F, molecule, pdb.topology

    def __init__(self, model_dict, name, smiles, path):
        self.lambda_electrostatics = [2.7e-7, 0.25, 0.5, 0.75, 1]
        self.lambda_sterics = [
            2.7e-7, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8,
            0.85, 0.9, 1
        ]
        self.n_steps = 10000
        self.report_interval = 500
        self._T = 300 * kelvin
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if (self.device == torch.device("cpu")):
            self.platform = Platform.getPlatformByName('CPU')
        else:
            self.platform = Platform.getPlatformByName('CUDA')

        self.model_dict = torch.load(model_dict, map_location=self.device)
        tot_unique = [
            0.14, 0.117, 0.155, 0.15, 0.21, 0.185, 0.18, 0.17, 0.12, 0.13
        ]
        self.model = GNN3_scale_96(max_num_neighbors=10000,
                                   parameters=None,
                                   device=self.device,
                                   fraction=0.5,
                                   unique_radii=tot_unique,
                                   jittable=True).to(self.device)
        self.model.load_state_dict(self.model_dict)
        self.model.to(self.device)
        self.model.eval()
        self.smiles = smiles
        self.name = name
        self.path = os.path.join(path, self.name)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.solv_path = os.path.join(self.path, f"{self.name}_solv")
        self.vac_path = os.path.join(self.path, f"{self.name}_vac")
        self.charges = None

    def set_model(self):
        assert self.system, self.solv_path

        cache_path = os.path.join(self.solv_path,
                                  f"{self.name}_gnn_paramed_model.pt")

        if not os.path.exists(cache_path):
            gnn_params = torch.tensor(self.compute_atom_features())

            self.model.gnn_params = gnn_params
            self.model.batch = torch.zeros(size=(len(gnn_params), )).to(torch.long)
            torch.jit.script(self.model).save(cache_path)

        self.model_force = TorchForce(cache_path)
        self.model_force.addGlobalParameter("lambda_sterics", 1.0)
        self.model_force.addGlobalParameter("lambda_electrostatics", 1.0)
        self.model_force.addGlobalParameter("retrieve_forces", 1.0)
        #self.model_force.addGlobalParameter("atom_features", 1.0)
        #self.model_force.addGlobalParameter('batch', -1.0)
        self.model_force.setOutputsForces(True)

    #taken from Michael
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

    def set_system(self, lambda_electrostatics):
        self.system, self.molecule, self.topology = self.create_system_from_smiles(
            self.smiles, self.pdb_path)
        print(" -- Saving Partial Charges -- ")
        nonbonded = [
            f for f in self.system.getForces()
            if isinstance(f, NonbondedForce)
        ][0]
        self.charges = np.array([
            tuple(nonbonded.getParticleParameters(idx))[0].value_in_unit(
                elementary_charge)
            for idx in range(self.system.getNumParticles())
        ])

    def savePDB(self, path):
        m = Chem.MolFromSmiles(self.smiles)
        mh = Chem.AddHs(m)
        AllChem.EmbedMolecule(mh)
        Chem.MolToPDBFile(mh, path)

    def compute_atom_features(self):
        '''Calculates the atom features needed for the GNN to function. the GB Force has derived parameters requiring 
        the derived function to pass through to calculate the radindex based on the GB radius
        '''
        atom_features_path = os.path.join(self.solv_path,
                                          f"{self.name}_gnn_params.pkl")

        if os.path.exists(atom_features_path):
            print("Found Existing Atom Features")
            with open(atom_features_path, 'rb') as f:
                data = pkl.load(f)

            return data.to(self.device)

        print("Calculating Atom Features for GNN")
        force = GBSAGBn2Force(cutoff=None,
                              SA="ACE",
                              soluteDielectric=1,
                              solventDielectric=78.5)
        print(78.5)
        gnn_params = np.array(force.getStandardParameters(self.topology))
        gnn_params = np.concatenate((np.reshape(self.charges,
                                                (-1, 1)), gnn_params),
                                    axis=1)
        force.addParticles(gnn_params)
        force.finalize()
        gbn2_parameters = np.array([
            force.getParticleParameters(i)
            for i in range(force.getNumParticles())
        ])
        print(f"Atom Features shape: {gbn2_parameters.shape}")

        with open(atom_features_path, 'wb') as f:
            pkl.dump(torch.from_numpy(gbn2_parameters), f)

        with open(atom_features_path, 'rb') as f:
            data = pkl.load(f)

        return data.to(self.device)

    def AI_simulation(self, lambda_sterics, lambda_electrostatics, vaccum,
                      out):
        assert self.vac_path
        assert self.solv_path

        if vaccum:
            gen_path = self.vac_path
        else:
            gen_path = self.solv_path
        path = os.path.join(gen_path, f"{out}.dcd")
        com = os.path.join(gen_path, f"{out}.com")

        if os.path.exists(com):
            return

        if vaccum:
            assert self.charges is not None
            new_charges = self.charges * lambda_electrostatics
            nbforces = self.forces['NonbondedForce']

            for idx in range(nbforces.getNumParticles()):
                charge, sigma, epsilon = nbforces.getParticleParameters(idx)
                nbforces.setParticleParameters(idx, new_charges[idx], sigma,
                                               epsilon)
        integrator = LangevinMiddleIntegrator(
            self._T, 1 / picosecond,
            0.002 * picoseconds)  #reduced from 2 femtoseconds temporarily
        simulation = Simulation(self.topology,
                                self.system,
                                integrator,
                                platform=self.platform)
        simulation.context.setParameter("lambda_sterics", lambda_sterics)
        simulation.context.setParameter("lambda_electrostatics",
                                        lambda_electrostatics)
        simulation.context.setParameter("retrieve_forces", 1.0)

        simulation.context.setPositions(self.PDB.positions)

        if vaccum:
            nbforces.updateParametersInContext(simulation.context)

        simulation.minimizeEnergy()
        simulation.reporters.append(DCDReporter(path, 100))

        simulation.step(self.n_steps)
        with open(com, 'w'):
            pass

    def calculate_energy_for_traj(self, traj, e_lambda_ster, e_lambda_elec):
        u = np.zeros(len(traj.time))
        e_lambda_ster = torch.scalar_tensor(e_lambda_ster).to(self.device)
        e_lambda_elec = torch.scalar_tensor(e_lambda_elec).to(self.device)

        for idx, coords in enumerate(traj.xyz):

            positions = torch.from_numpy(coords).to(self.device)
            batch = torch.zeros(size=(len(positions), )).to(torch.long) 

            factor = self.model(positions, e_lambda_ster, e_lambda_elec,
                                torch.tensor(0.0).to(self.device), batch, None)

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

    def solv_u_nk(self):
        cache_path = os.path.join(self.solv_path, f"{self.name}_u_nk.pkl")
        if os.path.exists(cache_path):
            return pd.read_pickle(cache_path)

        solv_u_nk_df = []

        self.set_system(0.0)
        integrator = LangevinIntegrator(self._T, 1 / picosecond,
                                        0.001 * picoseconds)
        self.curr_simulation_vac = Simulation(self.topology,
                                              self.system,
                                              integrator,
                                              platform=self.platform)
        # self.curr_simulation_vac.context.setParameter("vaccum", 1.0)

        for (lambda_ster,
             lambda_elec) in tqdm(self.get_solv_lambda_schedule()):
            dcd_file = os.path.join(
                self.solv_path,
                f"({lambda_ster}-{lambda_elec})_{self.name}.dcd")
            pdb_file = os.path.join(self.solv_path, f"{self.name}.pdb")
            traj = md.load(dcd_file, top=pdb_file)

            df = pd.DataFrame({
                "time": traj.time,
                "vdw-lambda": [lambda_ster] * len(traj.time),
                "coul-lambda": [lambda_elec] * len(traj.time),
            })
            df = df.set_index(["time", "vdw-lambda", "coul-lambda"])

            for (e_lambda_ster,
                 e_lambda_elec) in self.get_solv_lambda_schedule():
                u = self.calculate_energy_for_traj(traj, e_lambda_ster,
                                                   e_lambda_elec)
                df[(e_lambda_ster, e_lambda_elec)] = u

            df = self.u_nk_processing_df(df)

            solv_u_nk_df.append(df)

        solv_u_nk_df = alchemlyb.concat(solv_u_nk_df)

        #not sure what this does, is this just to add an extra index into the thing??
        new_index = []
        for i, index in enumerate(solv_u_nk_df.index):
            new_index.append((i, *index[1:]))
        solv_u_nk_df.index = pd.MultiIndex.from_tuples(
            new_index, names=solv_u_nk_df.index.names)

        solv_u_nk_df.to_pickle(cache_path)

        return solv_u_nk_df

    #Note to self: As the ML does not remove the intramolecular forces, readding the vaccum forces to MBAR is irrelevant. This function
    # is irrelevant.
    def vac_u_nk(self):
        cache_path = os.path.join(self.vac_path, f"{self.name}_u_nk.pkl")
        if os.path.exists(cache_path):
            return pd.read_pickle(cache_path)
        vac_u_nk_df = []

        for lambda_elec in self.lambda_electrostatics:
            dcd_file = os.path.join(self.vac_path,
                                    f"{lambda_elec}_{self.name}.dcd")
            pdb_file = os.path.join(self.solv_path, f"{self.name}.pdb")
            traj = md.load(dcd_file, top=pdb_file)

            df = pd.DataFrame({
                "time": traj.time,
                "fep_lambda": [lambda_elec] * len(traj.time)
            })

            df = df.set_index(["time", "fep_lambda"])

            for e_lambda_elec in self.lambda_electrostatics:
                u = self.calculate_energy_for_traj(traj, 0.0, e_lambda_elec)
                df[e_lambda_elec] = u

            df = self.u_nk_processing_df(df)

            vac_u_nk_df.append(df)

        vac_u_nk_df = alchemlyb.concat(vac_u_nk_df)

        new_index = []
        for i, index in enumerate(vac_u_nk_df.index):
            new_index.append((i, *index[1:]))
        vac_u_nk_df.index = pd.MultiIndex.from_tuples(
            new_index, names=vac_u_nk_df.index.names)

        vac_u_nk_df.to_pickle(cache_path)

        return vac_u_nk_df

    def run_all_sims(self, overwrite=False):
        print("-- Starting AI Simulation Hydration Calculations --")

        if overwrite:
            print("-- Overwriting Previous Records --")
            try:
                shutil.rmtree(self.solv_path)
                shutil.rmtree(self.vac_path)
            except:
                print(
                    "-- Files not Found / Nothing to Override; Continuing... --"
                )

        if not os.path.exists(self.solv_path):
            os.mkdir(self.solv_path)
        if not os.path.exists(self.vac_path):
            os.mkdir(self.vac_path)

        start = time.time()
        self.pdb_path = os.path.join(self.solv_path, f"{self.name}.pdb")
        if not os.path.exists(self.pdb_path):
            print("-- PDB of Smile not Found, Creating Now -- ")
            self.savePDB(
                self.pdb_path
            )  #ensure current system does not have changed electrostatics
        self.PDB = PDBFile(self.pdb_path)

        print("-- Setting System -- ")
        self.set_system(1.0)

        print("-- Setting Model -- ")
        self.set_model()

        print("-- Adding Model to System -- ")
        self.system.addForce(self.model_force)

        self.forces = {
            force.__class__.__name__: force
            for force in self.system.getForces()
        }
        setup_time = time.time()

        print(
            f" -- Finished Setup in {setup_time - start} seconds; Starting Simulation -- "
        )

        self.AI_simulation(1.0, 1.0, vaccum=0.0, out=f"(1.0-1.0)_{self.name}")

        print(" -- Removing Electrostatics -- ")
        for lambda_elec in reversed(self.lambda_electrostatics):
            self.AI_simulation(1.0,
                               lambda_elec,
                               vaccum=0.0,
                               out=f"(1.0-{lambda_elec})_{self.name}")
        solv_elec_time = time.time()
        print(
            f" -- Time taken: {solv_elec_time - setup_time} - Removing Sterics -- "
        )
        for lambda_ster in reversed(self.lambda_sterics):
            self.AI_simulation(lambda_ster,
                               0.0,
                               vaccum=0.0,
                               out=f"({lambda_ster}-0.0)_{self.name}")
        solv_ster_time = time.time()
        '''
        print(f" -- Time taken: {solv_ster_time - solv_elec_time} - Re-adding electrostatics in Vaccum -- ")
        for lambda_elec in self.lambda_electrostatics:
            self.AI_simulation(0.0, lambda_elec, vaccum = 1.0, out = f"{lambda_elec}_{self.name}")
        '''
        print(
            f" -- Finished Simulation -- Vaccum Time: {time.time() - solv_elec_time}; Total Time: {time.time() - start} -- "
        )

    def compute_delta_F(self):

        print(" -- Starting Calculation of Hydration Energy -- ")
        self.model.gnn_params = torch.tensor(self.compute_atom_feautres())  #idk why I am forced to do this but it helps for sum reason?
        solv = self.solv_u_nk()
        #vac = self.vac_u_nk()
        #mbar_vac = MBAR()
        #mbar_vac.fit(vac)

        mbar_solv = MBAR()
        mbar_solv.fit(solv)

        F_solv_kt = mbar_solv.delta_f_[(2.7e-07, 0.0)][(
            1.0, 1.0)]  #mbar_vac.delta_f_[0][1] -
        F_solv = F_solv_kt * self._T * kB

        return -F_solv.value_in_unit(kilojoule_per_mole) * 0.239006


class runSims:


    def __init__(self, model, path):
        self.model = model
        self.collect = []
        self.path = path

    def run_all_smiles(self, input):
        for val in input:

            smile = val[0]
            expt = val[1]
            print(f"Current Smile: {smile}, Expected Energy: {expt}")
            obj = AI_Solvation_calc(model_dict=self.model,
                                    smiles=smile,
                                    path=self.path,
                                    name="Initial")
            obj.run_all_sims(overwrite=False)
            res = obj.compute_delta_F()
            print(f"Calculated:{res}, Expected: {expt}")
            self.collect.append(res)
            del obj
            break


import sys

if __name__ == "__main__":

    model_path = '/work/users/r/d/rdey/ml_implicit_solvent/trained_models/280KDATASET5Kmodel.dict'

    smile = str(sys.argv[1])
    expt = float(sys.argv[2])
    name = str(sys.argv[3])
    path = '/work/users/r/d/rdey/test_check_MBAR'
    print(f"Current: {name}, {smile}, {expt}")
    obj = AI_Solvation_calc(model_dict=model_path,
                            smiles=smile,
                            path=path,
                            name=name)
    obj.run_all_sims(overwrite=False)
    res = obj.compute_delta_F()
    print(f"{name}, {res}, {expt}")
