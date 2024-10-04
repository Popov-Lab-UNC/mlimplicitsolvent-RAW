from copy import deepcopy
import os
import random
import pickle
import sys
from traceback import print_exc
import numpy as np
from tqdm import tqdm
import pandas as pd
#from config import CONFIG
from lr_complex import LRComplex, get_lr_complex
from openmm import unit, app
from openmm.app.dcdreporter import DCDReporter
from openmm.app.statedatareporter import StateDataReporter
import openmm as mm
import h5py
from rdkit import Chem
from rdkit.Chem import AllChem
from pymbar import timeseries
import math
import mdtraj as md
from fep import LAMBDA_STERICS, LAMBDA_ELECTROSTATICS, set_fep_lambdas
from freesolv import load_freesolv, smi_to_protonated_sdf
from sim import make_alchemical_system
import traceback
from scipy import stats




def get_parameter_derivative(simulation, param_name, dp=1e-5):
    """ 
    Uses finite difference to calculate the derivative of the 
    potential energy with respect to a parameter.
    """
    context = simulation.context
    parameter = context.getParameter(param_name)


    if (parameter == 1.0):
        dp = -dp

    initial_energy = context.getState(getEnergy=True).getPotentialEnergy()

    context.setParameter(param_name, parameter + dp)
    final_energy = context.getState(getEnergy=True).getPotentialEnergy()

    context.setParameter(param_name, parameter)

    return (final_energy - initial_energy) / dp


class EnergyReporter():
    def __init__(self, report_interval, vac_indicies):
        self.report_interval = report_interval
        self.energies = []
        self.forces = []
        self.sterics_derivative = []
        self.electrostatics_derivative = []
        self.vac_index = vac_indicies
    def describeNextReport(self, simulation):
        steps = self.report_interval - simulation.currentStep % self.report_interval
        return (steps, True, False, True, True, None)

    def report(self, simulation, state):
        sterics_derivative = get_parameter_derivative(simulation, LAMBDA_STERICS)
        electrostatics_derivative = get_parameter_derivative(simulation, LAMBDA_ELECTROSTATICS)
        potential_energy = state.getPotentialEnergy()
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)[self.vac_index]
        self.energies.append(potential_energy)
        self.forces.append(forces)
        self.electrostatics_derivative.append(electrostatics_derivative)
        self.sterics_derivative.append(sterics_derivative)

class SolvDatasetReporter():
    """ Saves forces, lambda derivatives, and positions
    (only for the ligand to save space) to an hdf5 file. """

    def __init__(self, filename, system_vac, report_interval):
        self.report_interval = report_interval
        self.system_vac = system_vac
        self.mol_indices = system_vac.lig_indices
        self.initial = False
        self.file = h5py.File(filename, 'w')
        for name in ("positions", "solv_forces"):
            self.file.create_dataset(name, maxshape=(None, None, 3), shape=(0, 0, 3), dtype=np.float32)

        for name in ("lambda_sterics", "lambda_electrostatics", "sterics_derivatives", "electrostatics_derivatives", "energies"):
            self.file.create_dataset(name, maxshape=(None,), shape=(0,), dtype=np.float32)

    def __del__(self):
        self.file.close()

    def describeNextReport(self, simulation):
        steps = self.report_interval - simulation.currentStep % self.report_interval
        return (steps, True, False, True, True, None)

    def report(self, simulation, state):
        
        if self.initial: #<- Only needs to be computed once due to MAFs
            self.positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            self.system_vac.set_positions(self.positions[self.mol_indices] * unit.nanometer)
            self.vac_forces = self.system_vac.get_forces().value_in_unit(unit.kilojoules_per_mole/unit.nanometer)
             # only compute electrostatics derivative; sterics derivative is zero
            self.vac_electrostatics_derivative = get_parameter_derivative(self.system_vac.simulation, LAMBDA_ELECTROSTATICS)
            self.initial = False


        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)
        U = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        sterics_derivative = get_parameter_derivative(simulation, LAMBDA_STERICS)
        electrostatics_derivative = get_parameter_derivative(simulation, LAMBDA_ELECTROSTATICS)

        # subtract off vac system forces and param derivatives so that we're
        # only looking at the ligand-solvent interactions
        lambda_sterics = simulation.context.getParameter(LAMBDA_STERICS)
        lambda_electrostatics = simulation.context.getParameter(LAMBDA_ELECTROSTATICS)

        self.file["positions"].resize((self.file["positions"].shape[0] + 1, len(self.mol_indices), 3))
        self.file["positions"][-1] = self.positions[self.mol_indices]

        self.file["solv_forces"].resize((self.file["solv_forces"].shape[0] + 1, len(self.mol_indices), 3))
        self.file["solv_forces"][-1] = forces[self.mol_indices] - self.vac_forces

        self.file["sterics_derivatives"].resize((self.file["sterics_derivatives"].shape[0] + 1,))
        self.file["sterics_derivatives"][-1] = sterics_derivative.value_in_unit(unit.kilojoules_per_mole)

        self.file["electrostatics_derivatives"].resize((self.file["electrostatics_derivatives"].shape[0] + 1,))
        self.file["electrostatics_derivatives"][-1] = (electrostatics_derivative - self.vac_electrostatics_derivative).value_in_unit(unit.kilojoules_per_mole)

        self.file["energies"].resize((self.file["energies"].shape[0] + 1,))
        self.file["energies"][-1] = U

        self.file["lambda_sterics"].resize((self.file["lambda_sterics"].shape[0] + 1,))
        self.file["lambda_sterics"][-1] = lambda_sterics

        self.file["lambda_electrostatics"].resize((self.file["lambda_electrostatics"].shape[0] + 1,))
        self.file["lambda_electrostatics"][-1] = lambda_electrostatics



def simulate_row(row):

    out_folder = os.path.join("/work/users/r/d/rdey/BigBindDataset", str(row.bigbind_index))
    os.makedirs(out_folder, exist_ok=True)

    out_file = out_folder + "/sim.h5"
    if os.path.exists(out_file):
        print(f"Already ran {row.lig_smiles}")
        return

    lig_file = os.path.join(out_folder, "ligand.sdf")
    if not os.path.exists(lig_file):
        smi_to_protonated_sdf(row.lig_smiles, lig_file)

    kwargs = {
        "nonbonded_cutoff": 0.9*unit.nanometer,
        # "nonbonded_cutoff": 1.5*unit.nanometer,
        "constraints": app.HBonds,
        "box_padding": 1.6*unit.nanometer,
        # "box_padding": 2.0*unit.nanometer,
        "lig_ff": "gaff",
        "cache_dir": out_folder,
    }

    system = get_lr_complex(None, lig_file,
                solvent="tip3p",
                nonbonded_method=app.PME,
                include_barostat=True,
                **kwargs
    )
    system_vac = get_lr_complex(None, lig_file,
                solvent="none",
                **kwargs
    )
    system.save(os.path.join(out_folder, "system"))
    system_vac.save(os.path.join(out_folder, "system_vac"))

    system.save_to_pdb(os.path.join(out_folder, "system.pdb"))
    system_vac.save_to_pdb(os.path.join(out_folder, "system_vac.pdb"))


    #FOR THE MAFs
    vac_indicies = system.lig_indices
    for i in vac_indicies:
        system.system.setParticleMass(i, 0.0) #only janky solution


    alc_system = make_alchemical_system(system)
    alc_system_vac = make_alchemical_system(system_vac)

    alc_system.set_positions(system.get_positions())

    full_frac = 0.2 # what fraction of the simulations we run with full interactions 

    if random.random() < full_frac:
        lambda_sterics = 1.0
        lambda_electrostatics = 1.0
    else:
        # alwaays remove electrostatics before sterics
        if random.random() < 0.5:
            lambda_sterics = random.uniform(0.0, 1.0)
            lambda_electrostatics = 0.0
        else:
            lambda_sterics = 1.0
            lambda_electrostatics = random.uniform(0.0, 1.0)

    steps = 200000

    print(f"Simulating {row.lig_smiles} for {steps} steps")
    print(f"lambda_sterics: {lambda_sterics:0.3f}, lambda_electrostatics: {lambda_electrostatics:0.3f}")


    set_fep_lambdas(alc_system.simulation.context, lambda_sterics, lambda_electrostatics)
    set_fep_lambdas(alc_system_vac.simulation.context, lambda_sterics, lambda_electrostatics)

    simulation = alc_system.simulation
    simulation.reporters = []

    try:
        os.remove(out_file)
    except FileNotFoundError:
        pass

    simulation.minimizeEnergy()

    reporter = SolvDatasetReporter(out_file, alc_system_vac, 500)
    simulation.reporters.append(reporter)
    
    



def simulate_MAF_row(row):
    out_folder = os.path.join("/work/users/r/d/rdey/ml_implicit_solvent/bigbind_solv/trials", str(row.bigbind_index)+"_MAF")
    os.makedirs(out_folder, exist_ok=True)
    MAF_out_file = out_folder + "/MAFsim.h5"
    out_file = out_folder + "/sim.h5"


    if os.path.exists(MAF_out_file):
        print(f"Already ran {row.lig_smiles}")
        return
    lig_file = os.path.join(out_folder, "ligand.sdf")
    if not os.path.exists(lig_file):
        smi_to_protonated_sdf(row.lig_smiles, lig_file)

    kwargs = {
        "nonbonded_cutoff": 0.9*unit.nanometer,
        # "nonbonded_cutoff": 1.5*unit.nanometer,
        "constraints": app.HBonds,
        "box_padding": 1.6*unit.nanometer,
        # "box_padding": 2.0*unit.nanometer,
        "lig_ff": "gaff",
        "cache_dir": out_folder,
    }
    system = get_lr_complex(None, lig_file,
                solvent="tip3p",
                nonbonded_method=app.PME,
                include_barostat=True,
                **kwargs
    )
    system_vac = get_lr_complex(None, lig_file,
                solvent="none",
                **kwargs
    )
    system.save(os.path.join(out_folder, "system"))
    system_vac.save(os.path.join(out_folder, "system_vac"))

    system.save_to_pdb(os.path.join(out_folder, "system.pdb"))
    system_vac.save_to_pdb(os.path.join(out_folder, "system_vac.pdb"))

    

    alc_system = make_alchemical_system(system)
    alc_system_vac = make_alchemical_system(system_vac)

    alc_system.set_positions(system.get_positions())

    

    full_frac = 0.5

    if random.random() < full_frac:
        lambda_sterics = 1.0
        lambda_electrostatics = 1.0
    else:
        # alwaays remove electrostatics before sterics
        if random.random() < 0.5:
            lambda_sterics = random.uniform(0.0, 1.0)
            lambda_electrostatics = 0.0
        else:
            lambda_sterics = 1.0
            lambda_electrostatics = random.uniform(0.0, 1.0)

    steps = 50000

    lambda_electrostatics = 0.8
    lambda_sterics = 1.0

    print(f"Simulating MAF Intial - {row.lig_smiles} for {steps} steps")
    print(f"lambda_sterics: {lambda_sterics:0.3f}, lambda_electrostatics: {lambda_electrostatics:0.3f}")

    set_fep_lambdas(alc_system.simulation.context, lambda_sterics, lambda_electrostatics)
    set_fep_lambdas(alc_system_vac.simulation.context, lambda_sterics, lambda_electrostatics)

    simulation = alc_system.simulation
    simulation.reporters = []

    try:
        os.remove(MAF_out_file)
    except FileNotFoundError:
        pass

    simulation.minimizeEnergy()
    dcd_file = os.path.join(out_folder, "simulation.dcd")
    MAFreporter = DCDReporter(file=dcd_file, reportInterval = 500)
    reporter = SolvDatasetReporter(out_file, alc_system_vac, 500)
    simulation.reporters.append(reporter)
    simulation.reporters.append(MAFreporter)
    simulation.step(steps)


    print(f"Simulating MAF Trajectories - {row.lig_smiles} for {steps} steps")

    traj = md.load(dcd_file, top = os.path.join(out_folder, "system.pdb"))
    df = pd.DataFrame({
                "name": [row.bigbind_index]*len(traj.time),
                "trajectory_time": traj.time,    
            })
    df = df.set_index(["name", "trajectory_time"])

    vac_indicies = system.lig_indices
    for i in vac_indicies:
        system.system.setParticleMass(i, 0.0) #only janky solution
    print(traj)
    
    solv_forces = []
    positions = []
    sterics_derivatives = []
    electrostatics_derivatives = []
    count = 0
    import time
    for coords in traj.xyz:
        start = time.time()
        reporter = EnergyReporter(500, vac_indicies)
        MAF_alc_system = make_alchemical_system(system)
        MAF_alc_system.set_positions(coords * unit.nanometer)
        set_fep_lambdas(MAF_alc_system.simulation.context, lambda_sterics, lambda_electrostatics)
        simulation = MAF_alc_system.simulation
        simulation.reporters.append(reporter)
        dcd = DCDReporter(file = '/work/users/r/d/rdey/ml_implicit_solvent/bigbind_solv/trials/old.dcd', reportInterval=500)
        simulation.reporters.append(dcd)
        simulation.step(200000)
        
        #set vaccum positions
        vac_coords = np.array(coords)[vac_indicies]
        alc_system_vac.set_positions(vac_coords.tolist()*unit.nanometer)
        vac_forces = alc_system_vac.get_forces().value_in_unit(unit.kilojoules_per_mole/unit.nanometer)

        U = reporter.energies
        forces = np.array(reporter.forces)
        dSterics = np.array(reporter.sterics_derivative)
        dElec = np.array(reporter.electrostatics_derivative)
        dElec_vac = get_parameter_derivative(alc_system_vac.simulation, LAMBDA_ELECTROSTATICS)
        dSter_vac = get_parameter_derivative(alc_system_vac.simulation, LAMBDA_STERICS)
    

        solv_force = np.nanmean(forces, axis=0) - np.array(vac_forces)
        solv_forces.append(solv_force)
        positions.append(vac_coords)
        electrostatics_derivatives.append(np.mean(dElec) - dElec_vac)
        sterics_derivatives.append(np.mean(dSterics) - dSter_vac)
        print(solv_forces[-1],
        positions[-1],
        sterics_derivatives[-1],
        electrostatics_derivatives[-1],)
        print(f"Count: {count} Time: {time.time() - start}")
        count +=1
        quit()
    report_MAF(df, positions, solv_forces, electrostatics_derivatives, sterics_derivatives, lambda_electrostatics, lambda_sterics, MAF_out_file, system_vac.lig_indices)


def report_MAF(df, positions, solv_forces, electrostatics_derivatives, sterics_derivatives, lambda_electrostatics, lambda_sterics, file_name, atom_indices):
    positions = iter(positions)
    solv_forces = iter(solv_forces)
    electrostatics_derivatives = iter(electrostatics_derivatives)
    sterics_derivatives = iter(sterics_derivatives)

    with h5py.File(file_name, 'a') as file:
        for name in ("positions", "solv_forces"):
            file.create_dataset(name, maxshape=(None, len(atom_indices), 3), shape=(0, len(atom_indices), 3), dtype=np.float32)
        for name in ("lambda_sterics", "lambda_electrostatics", "sterics_derivatives", "electrostatics_derivatives", "energies"):
            file.create_dataset(name, maxshape=(None,), shape=(0,), dtype=np.float32)

        for _, row in df.iterrows():
            file["positions"].resize((file["positions"].shape[0] + 1, len(atom_indices), 3))
            file["positions"][-1] = next(positions)

            file["solv_forces"].resize((file["solv_forces"].shape[0] + 1, len(atom_indices), 3))
            file["solv_forces"][-1] = next(solv_forces)

            file["sterics_derivatives"].resize((file["sterics_derivatives"].shape[0] + 1,))
            file["sterics_derivatives"][-1] = next(sterics_derivatives)

            file["electrostatics_derivatives"].resize((file["electrostatics_derivatives"].shape[0] + 1,))
            file["electrostatics_derivatives"][-1] = next(electrostatics_derivatives)

            file["lambda_sterics"].resize((file["lambda_sterics"].shape[0] + 1,))
            file["lambda_sterics"][-1] = lambda_sterics

            file["lambda_electrostatics"].resize((file["lambda_electrostatics"].shape[0] + 1,))
            file["lambda_electrostatics"][-1] = lambda_electrostatics

    


def simulate_slice(df, start_index, end_index):
    """ Simulate everything within this current slice """
    cur_df = df.iloc[start_index: end_index]
    #cur_df = df.iloc[(index*len(df))//total_indices:((index+1)*len(df))//total_indices]
    for i, row in tqdm(cur_df.iterrows(), total=len(cur_df)):
        try:
            simulate_row(row)
        except KeyboardInterrupt:
            raise
        except:
            print_exc()

def get_morgan_fps(smiles, radius=3, bits=2048):
    """ Get morgan fingerprints for each smiles, skipping ones that fail to parse"""
    fps = []
    for smi in tqdm(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=radius, nBits=bits)
        fps.append(np.array(fp, dtype=bool))
    fps = np.asarray(fps)
    return fps

def batch_tanimoto(fp, fps):
    """ Tanimoto similarity of fp to each of fps"""
    inter = np.logical_and(fp, fps)
    union = np.logical_or(fp, fps)
    sims = inter.sum(-1)/union.sum(-1)
    return sims

def get_splits(df):
    """ Returns a list of string splits ("val", "train", or "test") for
    each molecule in the dataframe. Split is "test" if the molecule is
    < 0.3 tanimoto similarity to anything in FreeSolv. Otherwise, split
    if val or test if the corresponding molecule is val or test in BigBind.
    Otherwise, split is randomly set to val or test 10% of the time """

    tan_cutoff = 0.3
    freesolv = load_freesolv()
    freesolv_fps = get_morgan_fps(freesolv.smiles)

    bb_fp_cache = os.path.join('/work/users/r/d/rdey/ml_implicit_solvent/bigbind_solv/trials', f"bigbind_solv", "bigbind_fps.npy")

    try:
        with open(bb_fp_cache, "rb") as f:
            bb_fps = np.load(f)
    except FileNotFoundError:
        bb_fps = get_morgan_fps(df.lig_smiles)
        with open(bb_fp_cache, "wb") as f:
            np.save(f, bb_fps)

    # none of the BigBinds mols should have failed
    assert len(bb_fps) == len(df)

    # compute tanimoto similarity from everything to everything in freesolv
    edges_cache = os.path.join('/work/users/r/d/rdey/ml_implicit_solvent/bigbind_solv/trials', f"bigbind_solv", f"bigbind_fp_edges_{tan_cutoff}.npy")

    try:
        with open(edges_cache, "rb") as f:
            edges = np.load(edges_cache)
    except FileNotFoundError:
        sims = []
        for fp in tqdm(freesolv_fps):
            sims.append(batch_tanimoto(fp, bb_fps))
        sims = np.array(sims)

        tan_cutoff = 0.3
        mask = sims > tan_cutoff
        edges = np.argwhere(mask)
        with open(edges_cache, "wb") as f:
            np.save(f, edges)
    
    # the indices of the molecules close to freesolv
    bb_freesolv_indices = set(df.bigbind_index[edges[:,1]])

    # now assign splits based on BigBind splits

    bb_split_smiles = {}
    for split in ["val", "test"]:
        bb_split_df = pd.read_csv(os.path.join(CONFIG.bigbind_dir, "activities_val.csv"))
        bb_split_smiles[split] = set(bb_split_df.lig_smiles)

    splits = []
    # make sure approx 10% of the datapoints are in test and val
    for i, row in tqdm(df.iterrows(), total=len(df)):
        split = "train"
        if row.bigbind_index in bb_freesolv_indices:
            split = "test"
        elif row.lig_smiles in bb_split_smiles["val"]:
            split = "val"
        elif row.lig_smiles in bb_split_smiles["test"]:
            split = "test"
        elif i % 10 == 0:
            split = "val"
        elif i % 10 == 1:
            split = "test"
        splits.append(split)

    return splits

def collate_dataset(df):
    """ Compiles all the data into train, val, and test h5 files in the
    CONFIG.bigbind_solv_dir folder. This filters out bad values and subsamples
    data using pymbar's decorrelation """

    np.seterr(all='raise')

    split_cache = os.path.join(CONFIG.cache_dir, f"bigbind_solv", "splits.pkl")
    try:
        with open(split_cache, "rb") as f:
            splits = pickle.load(f)
    except (FileNotFoundError, EOFError):
        splits = get_splits(df)
        with open(split_cache, "wb") as f:
            pickle.dump(splits, f)

    df["split"] = splits

    # store seperate hdf5 file for each split
    split_data = {}
    for split in df.split.unique():
        fname = os.path.join(CONFIG.bigbind_solv_dir, split + ".h5")
        if os.path.exists(fname):
            os.remove(fname)
        split_data[split] = h5py.File(fname, "w")

    for i, row in tqdm(df.iterrows(), total=len(df)):

        try:
            out_folder = os.path.join('/work/users/r/d/rdey/BigBindDataset', str(row.bigbind_index))
            out_file = out_folder + "/sim.h5"

            min_start_frame = 10
            
            file = h5py.File(out_file, 'r')

            # only save non-NaN stuff
            for key in file.keys():
                if np.isnan(file[key]).sum() > 0:
                    raise RuntimeError("Found NaNs!")

            if np.any(np.abs(file["solv_forces"]) > 10000):
                raise RuntimeError("Forces are too large")

            Us = file["energies"][:]
            t, g, Neff = timeseries.detect_equilibration(Us)

            t = max(t, min_start_frame)
            if len(Us[t::math.ceil(g)]) == 0:
                raise RuntimeError("Found no frames")

            data = split_data[row.split]
            group = data.create_group(str(row.bigbind_index))

            for key in file.keys():
                group[key] = file[key][t::math.ceil(g)]

            # now get the atomic numbers and charges from the OpenMM system
            # this should be all we need for the neural networks

            sys_prefix = out_folder + "/system_vac"
            complex = LRComplex.load(sys_prefix)

            atomic_numbers = np.array([ atom.element.atomic_number for atom in complex.topology.atoms() ])

            for force in complex.system.getForces():
                if isinstance(force, mm.NonbondedForce):
                    break
            else:
                raise RuntimeError("Failed to find Nonbonded force")

            charges = []
            for i in range(complex.system.getNumParticles()):
                q, *_ = force.getParticleParameters(i)
                charges.append(q.value_in_unit(unit.elementary_charge))
            charges = np.array(charges, dtype=np.float32)

            assert len(charges) == len(atomic_numbers)

            group["atomic_numbers"] = atomic_numbers
            group["charges"] = charges

            # now get the GB forces and energies...

            lig_file = os.path.join(out_folder, "ligand.sdf")

            kwargs = {
                "nonbonded_cutoff": 0.9*unit.nanometer,
                # "nonbonded_cutoff": 1.5*unit.nanometer,
                "constraints": app.HBonds,
                "box_padding": 1.6*unit.nanometer,
                # "box_padding": 2.0*unit.nanometer,
                "lig_ff": "espaloma",
                "cache_dir": out_folder,
            }
            complex_obc = get_lr_complex(None, lig_file,
                        solvent="obc2",
                        **kwargs
            )
            

        except FileNotFoundError:
            pass
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()

if __name__ == "__main__":
    df = pd.read_csv('/work/users/r/d/rdey/BigBindDataset/bigbind_diverse.csv')
    
    action = sys.argv[1]
    

    if action == "simulate":
        start_index = int(sys.argv[2])
        end_index = int(sys.argv[3])
        #total_indices = int(sys.argv[2])
        #index = int(sys.argv[3])
        simulate_slice(df, start_index, end_index)
    elif action == "collate":
        collate_dataset(df)
    else:
        print(f"Invalid action {action}")
