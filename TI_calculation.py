from MachineLearning.GNN_Models import GNN3_scale_96
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from datasets.bigbind_solv import MAFBigBind
from tqdm import tqdm
import os
import shutil
import alchemlyb.preprocessing
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator
from openmmtorch import TorchForce
from openmm.app.internal.customgbforces import GBSAGBn2Force
import torch
from openmm import app, NonbondedForce, LangevinMiddleIntegrator, Platform, LangevinIntegrator, MonteCarloBarostat
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


class AI_Solvation_calc_TI:

    @staticmethod
    def create_system_from_smiles(smiles, pdb_path):
        molecule = Molecule.from_smiles(smiles)
        pdb = app.PDBFile(pdb_path)
        smirnoff = GAFFTemplateGenerator(molecules=molecule)
        forcefield = app.ForceField()
        forcefield.registerTemplateGenerator(smirnoff.generator)
        system_F = forcefield.createSystem(topology=pdb.topology,
                                           nonbondedCutoff=0.9 * nanometer,
                                           constraints=app.HBonds)
        #system_F.addForce(MonteCarloBarostat(1*atmosphere, 300*kelvin))
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
        self.minimum_equil_frame = 5
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if (self.device == torch.device("cpu")):
            self.platform = Platform.getPlatformByName('CPU')
        else:
            self.platform = Platform.getPlatformByName('CUDA')

        print("Platform being used:", self.platform.getName())

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
        print(f"Net Charge: {round(sum(self.charges))}")

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
        integrator = LangevinMiddleIntegrator(self._T, 1 / picosecond,
                                              0.002 * picoseconds)
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

        print(
            f" -- Finished Simulation -- Total Time: {time.time() - start} -- "
        )

    def meanCalculation(self, traj, lambda_ster, lambda_elec):
        lambda_ster = torch.scalar_tensor(lambda_ster).to(self.device)
        lambda_elec = torch.scalar_tensor(lambda_elec).to(self.device)
        sterics = []
        electrostatics = []

        traj.center_coordinates()
        ref = traj[0]
        aligned = traj.superpose(reference=ref)
        avg_coords = np.mean(aligned.xyz, axis=0)
        avg_structure = md.Trajectory(xyz=avg_coords.reshape(1, -1, 3),
                                      topology=traj.topology)

        rmsd_vals = md.rmsd(target=traj,
                            reference=avg_structure,
                            frame=0,
                            atom_indices=None,
                            precentered=True)

        rmsd_mean = np.mean(rmsd_vals)
        rmsd_std = np.std(rmsd_vals)

        start_index = next((index for index, rmsd in enumerate(rmsd_vals)
                            if rmsd < (rmsd_mean + rmsd_std)),
                           self.minimum_equil_frame)

        start_index = self.minimum_equil_frame if start_index < self.minimum_equil_frame else start_index

        print(start_index)

        gnn_params = torch.cat((
            torch.tensor(self.compute_atom_features()),
            torch.full((len(avg_structure.xyz[0]), 1), lambda_elec),
            torch.full((len(avg_structure.xyz[0]), 1), lambda_ster),
        ),
                               dim=-1)
        batch = torch.zeros(size=(len(gnn_params), )).to(torch.long)                   
        '''
        gnn_params = gnn_params.repeat(len(traj[start_index:]))

        batch = torch.arange(0, len(traj[start_index:]))
        batch = batch.repeat_interleave(len(traj[0].xyz))

        positions = torch.from_numpy(traj[start_index:].xyz).float().reshape(-1, 3)

        lambda_elecs = torch.full((len(traj[start_index:]), 1), lambda_elec)
        lambda_sters = torch.full((len(traj[start_index:]), 1), lambda_ster)

        _, _, sterics, electrostatics = self.model(positions, lambda_sters, lambda_elecs, torch.tensor(0.0), True, batch, gnn_params)

        return (lambda_elec.item(),
                (torch.mean(torch.tensor(electrostatics).detach()).item()),
                lambda_ster.item(),
                (torch.mean(torch.tensor(sterics).detach()).item()))
        
        '''
        for idx, coords in enumerate(traj[start_index:].xyz):
            positions = torch.from_numpy(coords).to(self.device)
            positions = positions.float()
            lambda_ster = lambda_ster.float()
            lambda_elec = lambda_elec.float()
            U, F, steric, electrostatic = self.model(
                positions, lambda_ster, lambda_elec,
                torch.tensor(0.0).to(self.device), True, batch, gnn_params)
            
            sterics.append(steric)
            electrostatics.append(electrostatic)
        return (lambda_elec.item(),
                (torch.mean(torch.tensor(electrostatics).detach()).item()),
                lambda_ster.item(),
                (torch.mean(torch.tensor(sterics).detach()).item()))
        #'''
    def collateInfo(self):
        derivatives = []
        self.set_model()
        for (lambda_ster,
             lambda_elec) in tqdm(self.get_solv_lambda_schedule()):
            dcd_file = os.path.join(
                self.solv_path,
                f"({lambda_ster}-{lambda_elec})_{self.name}.dcd")
            pdb_file = os.path.join(self.solv_path, f"{self.name}.pdb")
            traj = md.load(dcd_file, top=pdb_file)
            derivatives.append(
                self.meanCalculation(traj, lambda_ster, lambda_elec))
        print(derivatives)
        derivatives = np.array(derivatives)
        elec = derivatives[:, 0]
        elec_der = derivatives[:, 1]
        ster = derivatives[:, 2]
        ster_der = derivatives[:, 3]

        np_elec = np.array(elec)
        np_elec_der = np.array(elec_der)
        np_ster = np.array(ster)
        np_ster_der = np.array(ster_der)

        u_elec = np.unique(np_elec)
        u_ster = np.unique(np_ster)

        u_elec_der = np.array(
            [np.median(np_elec_der[np_elec == ux]) for ux in u_elec])
        u_ster_der = np.array(
            [np.median(np_ster_der[np_ster == ux]) for ux in u_ster])

        sort_elec = np.argsort(u_elec)
        sort_ster = np.argsort(u_ster)

        int_elec = np.trapz(u_elec_der[sort_elec], u_elec[sort_elec])
        int_ster = np.trapz(u_ster_der[sort_ster], u_ster[sort_ster])

        return ((int_elec + int_ster) / 4.1), derivatives


if __name__ == "__main__":

    model_path = '/work/users/r/d/rdey/ml_implicit_solvent/trained_models/280KDATASET5Kmodel.dict'

    smile = str(sys.argv[1])
    expt = float(sys.argv[2])
    name = str(sys.argv[3])
    path = '/work/users/r/d/rdey/test_check_500'
    print(f"Current: {name}, {smile}, {expt}")
    obj = AI_Solvation_calc_TI(model_dict=model_path,
                               smiles=smile,
                               path=path,
                               name=name)
    obj.run_all_sims(overwrite=False)
    res, _ = obj.collateInfo()
    print(f"{name}, {res}, {expt}")
