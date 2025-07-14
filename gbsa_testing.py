from openmm.unit import *
import torch
from openmm import app, NonbondedForce, Platform, LangevinIntegrator
from openmm.app import *
from openmm.app.internal.customgbforces import GBSAGBn2Force
from MachineLearning.GNN_Models import GNN3_scale_96
import os
import parmed as pmd
import numpy as np
from copy import deepcopy
import warnings
import alchemlyb.preprocessing
import mdtraj as md
import pandas as pd
from tqdm import tqdm
from openmmtools.constants import kB
from alchemlyb.estimators import MBAR
import pickle as pkl
import sys

class MM_GBSA():
  """
  Preliminary MM_GBSA Calculations completed on simulations done beforehand utilizing GBn2. For testing purposes only. 
  """

  def __init__(self, top_path, dcd_path, name, model_dict, path):

    print(f"Running MM-GBSA Calculation on: {name}")

    self.lambda_electrostatics = [0.0, 1.0]
    self.lambda_sterics = [0.0, 1.0]

    self._T = 300 * kelvin

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (self.device == torch.device("cpu")):
      self.platform = Platform.getPlatformByName('CPU')
    else:
      self.platform = Platform.getPlatformByName('CUDA')

    print("Loading Model... ")
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
    self.name = name
    self.path = os.path.join(path, name)
    if not os.path.exists(self.path):
            os.mkdir(self.path)


    #Initializing needed variables. 

    print("Loading Topologies and Trajectories... ")

    self.lr_top = self.get_lr_topology(top_path=top_path)

    print(f"Total Atoms in lr_top: {self.lr_top.topology.getNumAtoms()}")

    mdtraj_top = md.Topology.from_openmm(self.lr_top.topology)

    self.lr_traj = md.load(dcd_path, top = mdtraj_top)

    

    self.lig_indices = self.get_lig_indices(self.lr_top)
    self.rec_indices = np.setdiff1d(np.arange(self.lr_top.topology.getNumAtoms()), self.lig_indices)

    self.rec_traj = self.lr_traj.atom_slice(self.rec_indices)
    self.lig_traj = self.lr_traj.atom_slice(self.lig_indices)  

    print("Loading Systems... ")
    self.lr_system = self.get_lr_system()
    self.rec_system, self.rec_top = self.get_rec_system()
    self.lig_system, self.lig_top = self.get_lig_system()

    print("Computing Atom Features for Model... ")
    self.lr_params = self.compute_atom_features('lr', self.lr_top.topology, self.lr_system)
    self.rec_params = self.compute_atom_features('rec', self.rec_top.topology, self.rec_system)
    self.lig_params = self.compute_atom_features('lig', self.lig_top.topology, self.lig_system)



  def get_lr_topology(self, top_path):
    top_file = f"{top_path}/HMR_complex.parm7"
    top = pmd.load_file(top_file)
    top.strip(":WAT,SOL,HOH,Na+,Cl-,NA,CL,")
    pmd.tools.actions.changeRadii(top, "mbondi2").execute()
    return top
  
  def get_lig_indices(self, top):
    lig_res = None
    for res in top.residues:
      if res.name == "LIG":
        lig_res = res
        break
    return np.array([atom.idx for atom in lig_res.atoms], dtype=int)
  
  def get_lr_system(self):
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = self.lr_top.createSystem(
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
            )
    return system
  
  def get_lig_system(self):
    top = deepcopy(self.lr_top)
    top.strip("!:LIG")
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = top.createSystem(
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
            )
    return system, top

  def get_rec_system(self):
    top = deepcopy(self.lr_top)
    top.strip(":LIG")
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = top.createSystem(
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
            )
    return system, top
  
  def compute_atom_features(self, tag, topology, system):


    cache_path = os.path.join(self.path, f"{tag}_gnn_params.pkl")

    if os.path.exists(cache_path):
      print("Found Existing Atom Features")
      with open(cache_path, 'rb') as f:
          data = pkl.load(f)
          return data.to(self.device)

    
    force = GBSAGBn2Force(cutoff=None,
                            SA="ACE",
                            soluteDielectric=1,
                            solventDielectric=78.5)
    gnn_params = np.array(force.getStandardParameters(topology))
    nonbonded = [
            f for f in system.getForces()
            if isinstance(f, NonbondedForce)
        ][0]
    charges = np.array([
          tuple(nonbonded.getParticleParameters(idx))[0].value_in_unit(
              elementary_charge)
          for idx in range(system.getNumParticles())
      ])
    gnn_params = np.concatenate((np.reshape(charges,
                                                (-1, 1)), gnn_params),
                                    axis=1)
    force.addParticles(gnn_params)
    force.finalize()

    gbn2_parameters = np.array([
            force.getParticleParameters(i)
            for i in range(force.getNumParticles())
        ])
    
    data = torch.from_numpy(gbn2_parameters).to(self.device)

    with open(cache_path, 'wb') as f:
      pkl.dump(data, f)

    return data.to(self.device)

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
  
  def u_nk_processing_df(self, df):
    df.attrs = {
        "temperature": self._T,
        "energy_unit": "kT",
    }

    df = alchemlyb.preprocessing.decorrelate_u_nk(df, remove_burnin=True)
    return df


  def calculateMMGBSA(self, tag, traj, topology, system, gnn_params):

    traj = traj[-300:]

    print(f"Calculating Energies matrix for {tag}.")    

    u = np.zeros(len(traj.time))

    e_lambda_ster = torch.scalar_tensor(1.0).to(self.device)
    e_lambda_elec = torch.scalar_tensor(1.0).to(self.device)

    integrator = LangevinIntegrator(self._T, 1 / picosecond,
                                        0.001 * picoseconds)
    curr_simulation_vac = Simulation(topology,
                                              system,
                                              integrator,
                                              platform=self.platform)

    for idx, coords in tqdm(enumerate(traj.xyz), total=len(traj.xyz)):

      positions = torch.from_numpy(coords).to(self.device)
      batch = torch.zeros(size=(len(positions), ), device=self.device).to(torch.long) 

      factor = self.model(positions, e_lambda_ster, e_lambda_elec,
                          torch.tensor(1.0).to(self.device), True, batch, gnn_params)
      curr_simulation_vac.context.setPositions(coords)

      U = curr_simulation_vac.context.getState(
                getEnergy=True).getPotentialEnergy()
      
      val = U.value_in_unit(kilojoule_per_mole) + factor[0].item()

      u[idx] = val

    U = torch.tensor(u, device=self.device)

    E_U = torch.mean(U)

    print(f"E_U for {tag}: {E_U}")

    return E_U

  def compute_G(self):

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=3) as executor:
      lig_future = executor.submit(self.calculateMMGBSA("lig", self.lig_traj, self.lig_top, self.lig_system, self.lig_params))
      lr_future = executor.submit(self.calculateMMGBSA("lr", self.lr_traj, self.lr_top, self.lr_system, self.lr_params))
      rec_future = executor.submit(self.calculateMMGBSA("rec", self.rec_traj, self.rec_top, self.rec_system, self.rec_params))

      lig_solv = lig_future.result()
      lr_solv = lr_future.result()
      rec_solv = rec_future.result()

    return lr_solv - (lig_solv + rec_solv)



if __name__ == "__main__":
  
  model_path = '/work/users/r/d/rdey/ml_implicit_solvent/trained_models/280KDATASET2Kv3model.dict'
  name = str(sys.argv[1])
  top_path = str(sys.argv[2])
  dcd_path = str(sys.argv[3])




  path = "/work/users/r/d/rdey/ml_implicit_solvent/gbsa_calculations"

  mmgbsa = MM_GBSA(top_path=top_path, dcd_path=dcd_path, name = name, model_dict=model_path, path = path)
  print(mmgbsa.compute_G())



  top_path = "/proj/kpoplab/fragment-opt-abfe-benchmark/topologies_and_structure/HSP90/ligand-15/complex/amber/"
  dcd_path = "/proj/kpoplab/dbfe_cache/dbfe_v4/fragment/HSP90/15/lr/traj.dcd"
