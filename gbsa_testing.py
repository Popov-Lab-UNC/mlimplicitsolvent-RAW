from openmm.unit import *
import torch
from openmm import app, NonbondedForce, LangevinMiddleIntegrator, Platform, LangevinIntegrator
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


class MM_GBSA():
  """
  Preliminary MM_GBSA Calculations completed on simulations done utilizing GBn2. For testing purposes only. 
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

    self.lr_traj = md.load(dcd_path, top = self.lr_top.topology)

    self.lig_indices = self.get_lig_indices(self.lr_top)
    self.rec_indices = np.setdiff1d(np.arange(self.lr_top.topology.getNumAtoms()), self.lig_indices)

    self.rec_traj = self.lr_traj.atom_slice(self.rec_indices)
    self.lig_traj = self.lr_traj.atom_slice(self.lig_indices)  

    print("Loading Systems... ")
    self.lr_system = self.get_lr_system()
    self.rec_system, self.rec_top = self.get_rec_system()
    self.lig_system, self.lig_top = self.get_lig_system()

    print("Computing Atom Features for Model... ")
    self.lr_params = self.compute_atom_features(self.lr_top, self.lr_system)
    self.rec_params = self.compute_atom_features(self.rec_top, self.rec_system)
    self.lig_params = self.compute_atom_features(self.lig_top, self.lig_system)

    self.curr_simulation_vac = None

  def get_lr_topology(self, top_path):
    top_file = f"{top_path}/HMR_complex.parm7"
    top = pmd.load_file(top_file)
    top.strip(":WAT,Na+,Cl-")
    pmd.tools.actions.changeRadii(top, "mbondi2").execute()
    return top
  
  def get_lig_indices(self, top):
    lig_res = None
    for res in top.residues:
      if res.name == "LIG":
        lig_res = res
        break
    return np.array([atom.index for atom in lig_res.atoms], dtype=int)
  
  def get_lr_system(self):
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = self.lr_top.createSystem(
                implicitSolvent=app.OBC2,
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
                useSASA=True
            )
    return system
  
  def get_lig_system(self):
    top = deepcopy(self.lr_top)
    top.strip(f":0-{top.residues[-1].number}")
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = top.createSystem(
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
                useSASA=True
            )
    return system, top

  def get_rec_system(self):
    top = deepcopy(self.lr_top)
    top.strip(f":{top.residues[-1].number+1}")
    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = top.createSystem(
                implicitSolvent=app.OBC2,
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
                useSASA=True
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
    
    data = torch.from_numpy(gbn2_parameters)

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
     
  def calculate_energy_for_traj(self, traj, e_lambda_ster, e_lambda_elec, gnn_params):
    u = np.zeros(len(traj.time))
    e_lambda_ster = torch.scalar_tensor(e_lambda_ster).to(self.device)
    e_lambda_elec = torch.scalar_tensor(e_lambda_elec).to(self.device)

    for idx, coords in enumerate(traj.xyz):

      positions = torch.from_numpy(coords).to(self.device)
      batch = torch.zeros(size=(len(positions), )).to(torch.long) 

      factor = self.model(positions, e_lambda_ster, e_lambda_elec,
                          torch.tensor(1.0).to(self.device), True, batch, gnn_params)

      self.curr_simulation_vac.context.setPositions(coords)
      self.curr_simulation_vac.minimizeEnergy()

      U = self.curr_simulation_vac.context.getState(
          getEnergy=True).getPotentialEnergy()
      val = (U + (factor[0].item() * kilojoule_per_mole)) / (kB * 300)
      u[idx] = float(val)
    return u

  def get_solv_u_nk(self, tag, traj, topology, system, gnn_params):

    print(f"Calculating MBAR matrix for {tag}.")

    cache_path = os.path.join(self.path, f"{tag}_u_nk.pkl")

    if os.path.exists(cache_path):
      print("Found Cache-- Continuing..." )
      return pd.read_pickle(cache_path)

    solv_u_nk_df = []

    integrator = LangevinIntegrator(self._T, 1 / picosecond,
                                        0.001 * picoseconds)

    self.curr_simulation_vac = Simulation(topology,system,
                                              integrator,
                                              platform=self.platform)
    for (lambda_ster, lambda_elec) in tqdm(self.get_solv_lambda_schedule()):
      df = pd.DataFrame({
          "time": traj.time,
          "vdw-lambda": [lambda_ster] * len(traj.time),
          "coul-lambda": [lambda_elec] * len(traj.time),
      })
      df = df.set_index(["time", "vdw-lambda", "coul-lambda"])

      for (e_lambda_ster, e_lambda_elec) in self.get_solv_lambda_schedule():

            u = self.calculate_energy_for_traj(traj, e_lambda_ster, e_lambda_elec, gnn_params)
            df[(e_lambda_ster, e_lambda_elec)] = u

      df = self.u_nk_processing_df(df)
      solv_u_nk_df.append(df)
    solv_u_nk_df = alchemlyb.concat(solv_u_nk_df)

    new_index = []
    for i, index in enumerate(solv_u_nk_df.index):
        new_index.append((i, *index[1:]))
    solv_u_nk_df.index = pd.MultiIndex.from_tuples(
        new_index, names=solv_u_nk_df.index.names)
    
    solv_u_nk_df.to_pickle(cache_path)

    return solv_u_nk_df
  
  def compute_G(self):
    lr_solv = self.get_solv_u_nk("lr", self.lr_traj, self.lr_top, self.lr_system, self.lr_params)
    lig_solv = self.get_solv_u_nk("lig", self.lig_traj, self.lig_top, self.lig_system, self.lig_params)
    rec_solv = self.get_solv_u_nk("rec", self.rec_traj, self.rec_top, self.rec_system, self.rec_params)

    mbar_lr_solv = MBAR()
    mbar_lig_solv = MBAR()
    mbar_rec_solv = MBAR()

    mbar_lr_solv.fit(lr_solv)
    mbar_lig_solv.fit(lig_solv)
    mbar_rec_solv.fit(rec_solv)

    F_lr_solv_kt = mbar_lr_solv.delta_f_[(0.0, 0.0)][(1.0, 1.0)] 
    F_lr_solv = F_lr_solv_kt * self._T * kB

    F_lig_solv_kt = mbar_lig_solv.delta_f_[(0.0, 0.0)][(1.0, 1.0)] 
    F_lig_solv = F_lig_solv_kt * self._T * kB

    F_rec_solv_kt = mbar_rec_solv.delta_f_[(0.0, 0.0)][(1.0, 1.0)] 
    F_rec_solv = F_rec_solv_kt * self._T * kB

    F_solv = F_lr_solv - (F_lig_solv + F_rec_solv)

    return -F_solv.value_in_unit(kilojoule_per_mole) * 0.239006



if __name__ == "__main__":
  model_path = '/work/users/r/d/rdey/ml_implicit_solvent/trained_models/280KDATASET2Kv3model.dict'
  top_path = "/proj/kpoplab/fragment-opt-abfe-benchmark/topologies_and_structure/HSP90/ligand-15/complex/amber/HMR_complex.parm7"
  dcd_path = "/proj/kpoplab/dbfe_cache/dbfe_v4/fragment/HSP90/15/lr/traj.dcd"
  path = "/work/users/r/d/rdey/ml_implicit_solvent/gbsa_calculations"

  mmgbsa = MM_GBSA(top_path=top_path, dcd_path=dcd_path, name = "HSP90-15", model_dict=model_path, path = path )

