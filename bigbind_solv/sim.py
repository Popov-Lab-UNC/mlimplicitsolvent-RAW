import copy
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import mdtraj as md
import openmm as mm
from openmm import app, NonbondedForce
from openmm import unit
from openmmtools.constants import kB
import alchemlyb
from alchemlyb.estimators import MBAR
from alchemlyb.preprocessing.subsampling import decorrelate_u_nk
from lr_complex import get_lr_complex
from fep import apply_fep, set_fep_lambdas
import pickle as pkl
from openmm.app.internal.customgbforces import GBSAGBn2Force

def get_lig_and_water_indices(system):
    """ Returns lists of sets of the ligand and water atom indices.
    This is the format needed by fep.py """

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


def make_alchemical_system(system):
    """ Created a new alchemically modified LR system"""
    lig_indices, water_indices = get_lig_and_water_indices(system)

    # create a new alchemical system
    alchemical_system = apply_fep(system.system, lig_indices, water_indices)

    properties = {'Precision': 'mixed', 'DeviceIndex': '0'}

    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin,
                                             1.0 / unit.picosecond,
                                             2.0 * unit.femtosecond)
    simulation = app.Simulation(system.topology, alchemical_system, integrator,
                                system.platform, properties)

    lr_system = copy.copy(system)
    lr_system.system = alchemical_system
    lr_system.integrator = integrator
    lr_system.simulation = simulation

    return lr_system


class SolvationSim:
    """ Class for running all the MD for the alchemical
    solvation free energy calculations. """

    def __init__(self, lig_file, out_folder):

        self.out_folder = out_folder
        os.makedirs(out_folder, exist_ok=True)

        kwargs = {
            "nonbonded_cutoff": 0.9 * unit.nanometer,
            # "nonbonded_cutoff": 1.5*unit.nanometer,
            "constraints": app.HBonds,
            "box_padding": 1.6 * unit.nanometer,
            # "box_padding": 2.0*unit.nanometer,
            "lig_ff": "gaff",
            "cache_dir": out_folder,
        }

        system = get_lr_complex(None,
                                lig_file,
                                solvent="none",
                                nonbonded_method=app.NoCutoff,
                                **kwargs)
        
        gbsa_force = GBSAGBn2Force(cutoff=None,
                                    SA="ACE",
                                    soluteDielectric=1,
                                    solventDielectric=78.5)
        
        self.gbsa_params = np.array(gbsa_force.getStandardParameters(system.topology))
        nonbonded = [
            f for f in system.system.getForces()
            if isinstance(f, NonbondedForce)
            ][0]
        
        self.charges = np.array([
            tuple(nonbonded.getParticleParameters(idx))[0].value_in_unit(
                unit.elementary_charge)
            for idx in range(system.system.getNumParticles())
        ])

        intial_sterics = np.repeat(1.0, len(self.charges))

        gbsa_params = np.concatenate(
                                    (np.reshape(self.charges, (-1, 1)), self.gbsa_params, np.reshape(intial_sterics, (-1, 1))),
                                    axis=1
                                    )
        
        gbsa_force.addParticles(gbsa_params)
        gbsa_force.finalize()
        system.system.addForce(gbsa_force)


        system_vac = get_lr_complex(None, lig_file, solvent="none", **kwargs)
        system.save(os.path.join(out_folder, "system"))
        system_vac.save(os.path.join(out_folder, "system_vac"))

        system.save_to_pdb(os.path.join(out_folder, "system.pdb"))
        system_vac.save_to_pdb(os.path.join(out_folder, "system_vac.pdb"))

        self.system = make_alchemical_system(system)

        


        self.system_vac = make_alchemical_system(system_vac)

        self.system.set_positions(system.get_positions())

        self.electrostatics_schedule = [
            0.00,
            0.50,
            1.00,
        ]
        self.sterics_schedule = [
            0.00,
            0.25,
            0.50,
            0.75,
            1.00,
        ]

        self.equil_steps = 25000

    def minimize(self):
        cache_file = os.path.join(self.out_folder, "minimized.pkl")
        tol = 1.0
        try:
            self.system.minimize_cached(cache_file, tol)
        except mm.OpenMMException:
            # in case we create a system with a different number of waters
            os.remove(cache_file)
            self.system.minimize_cached(cache_file, tol)

    def get_sim_prefix(self,
                       lambda_sterics,
                       lambda_electrostatics,
                       vacuum=False):
        """ Returns the prefix for the simulation file """
        prefix = f"sim_{lambda_sterics}_{lambda_electrostatics}"
        if vacuum:
            prefix += "_vac"
        return prefix

    def simulate(self,
                 n_steps,
                 lambda_sterics,
                 lambda_electrostatics,
                 prefix=None,
                 vacuum=False):
        """ Run the simulation for n_steps, saving the trajectory
        to out_folder/prefix.dcd """
        if prefix is None:
            prefix = self.get_sim_prefix(lambda_sterics, lambda_electrostatics,
                                         vacuum)
        out_dcd = os.path.join(self.out_folder, f"{prefix}.dcd")
        out_com = os.path.join(self.out_folder, f"{prefix}.completed")
        if os.path.exists(out_com):
            print(f"Skipping {prefix} -- already completed")
            return

        if vacuum:
            simulation = self.system_vac.simulation
        else:
            simulation = self.system.simulation
            
            new_charges = self.charges * lambda_electrostatics
            sterics = np.repeat(lambda_sterics, len(self.charges))

            gbsa_params = np.concatenate(
                                    (np.reshape(new_charges, (-1, 1)), self.gbsa_params, np.reshape(sterics, (-1, 1))),
                                    axis=1
                                    )

            gbsa_force = [
            f for f in simulation.system.getForces()
            if isinstance(f, mm.CustomGBForce)
            ][0]
            print(gbsa_force.getParticleParameters(0))

            for idx, params in enumerate(gbsa_params):
                gbsa_force.setParticleParameters(idx, params.tolist())
            gbsa_force.updateParametersInContext(simulation.context)
            

            gbsa_force = [
            f for f in simulation.system.getForces()
            if isinstance(f, mm.CustomGBForce)
            ][0]

            print(gbsa_force.getParticleParameters(0))
        
        
        set_fep_lambdas(simulation.context, lambda_sterics,
                        lambda_electrostatics)
        simulation.reporters.clear()
        simulation.reporters.append(app.DCDReporter(out_dcd, 100))
        simulation.step(n_steps)

        with open(out_com, "w") as f:
            f.write("completed")

    def get_vac_u_nk(self, lambda_elec, T=300 * unit.kelvin):
        """ Returns the u_nk dataframe for the vacuum simulation 
        run at lambda_elec. This returns the energy of the system
        from _all_ the lambda_elec values for the simulation """

        prefix = self.get_sim_prefix(0.0, lambda_elec, vacuum=True)
        cache_fname = os.path.join(self.out_folder, f"{prefix}_u_nk.pkl")
        if os.path.exists(cache_fname):
            return pd.read_pickle(cache_fname)

        dcd_file = prefix + ".dcd"
        dcd_file = os.path.join(self.out_folder, dcd_file)
        pdb_file = os.path.join(self.out_folder, "system_vac.pdb")

        traj = md.load(dcd_file, top=pdb_file)
        df = pd.DataFrame({
            "time": traj.time,
            "fep-lambda": [lambda_elec] * len(traj.time),
        })
        df = df.set_index(["time", "fep-lambda"])

        for energy_lambda_elec in self.electrostatics_schedule:
            u = np.zeros(len(traj.time))
            set_fep_lambdas(self.system_vac.simulation.context, 0.0,
                            energy_lambda_elec)
            for i, coords in enumerate(traj.xyz):
                self.system_vac.set_positions(coords * unit.nanometer)

                U = self.system_vac.simulation.context.getState(
                    getEnergy=True).getPotentialEnergy()
                # reduced energy (divided by kT)
                u[i] = U / (kB * T)
            df[energy_lambda_elec] = u

        df.attrs = {
            "temperature": T.value_in_unit(unit.kelvin),
            "energy_unit": "kT",
        }

        df = decorrelate_u_nk(df, remove_burnin=True)

        df.to_pickle(cache_fname)

        return df

    def get_all_vac_u_nk(self):
        df = alchemlyb.concat([
            self.get_vac_u_nk(lambda_elec)
            for lambda_elec in self.electrostatics_schedule
        ])
        # make sure time is increasing
        new_index = []
        for i, index in enumerate(df.index):
            new_index.append((i, *index[1:]))
        df.index = pd.MultiIndex.from_tuples(new_index, names=df.index.names)
        return df

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

    def get_solv_u_nk(self, lambda_ster, lambda_elec, T=300 * unit.kelvin):
        """ Returns the u_nk dataframe for the solvation simulation run
        at lambda_ster and lambda_elec. """

        prefix = self.get_sim_prefix(lambda_ster, lambda_elec)
        cache_fname = os.path.join(self.out_folder, f"{prefix}_u_nk.pkl")
        if os.path.exists(cache_fname):
            return pd.read_pickle(cache_fname)

        dcd_file = prefix + ".dcd"
        dcd_file = os.path.join(self.out_folder, dcd_file)
        pdb_file = os.path.join(self.out_folder, "system.pdb")

        traj = md.load(dcd_file, top=pdb_file)
        df = pd.DataFrame({
            "time": traj.time,
            "vdw-lambda": [lambda_ster] * len(traj.time),
            "coul-lambda": [lambda_elec] * len(traj.time),
        })
        df = df.set_index(["time", "vdw-lambda", "coul-lambda"])

        for energy_lambda_ster, energy_lambda_elec in self.get_solv_lambda_schedule(
        ):
            u = np.zeros(len(traj.time))
            set_fep_lambdas(self.system.simulation.context, energy_lambda_ster,
                            energy_lambda_elec)
            gbsa_force = [
            f for f in self.system.system.getForces()
            if isinstance(f, mm.CustomGBForce)
            ][0]


            


            
            for i, coords in enumerate(traj.xyz):
                

                self.system.set_positions(coords * unit.nanometer)

                new_charges = self.charges * energy_lambda_elec

                sterics = np.repeat(energy_lambda_ster, len(self.charges))

                gbsa_params = np.concatenate((np.reshape(new_charges, (-1, 1)), self.gbsa_params, np.reshape(sterics, (-1, 1))),
                                    axis=1)

                for idx, params in enumerate(gbsa_params):
                    gbsa_force.setParticleParameters(idx, params.tolist())
                    gbsa_force.updateParametersInContext(self.system.simulation.context)
                    
                U = self.system.simulation.context.getState(
                    getEnergy=True).getPotentialEnergy()

                # reduced energy (divided by kT)
                u[i] = U / (kB * T)
            df[(energy_lambda_ster, energy_lambda_elec)] = u

        df.attrs = {
            "temperature": T.value_in_unit(unit.kelvin),
            "energy_unit": "kT",
        }

        df = decorrelate_u_nk(df, remove_burnin=True)

        df.to_pickle(cache_fname)

        return df

    def get_all_solv_u_nk(self):
        df = alchemlyb.concat([
            self.get_solv_u_nk(lambda_ster, lambda_elec) for lambda_ster,
            lambda_elec in tqdm(self.get_solv_lambda_schedule())
        ])
        # make sure time is increasing
        new_index = []
        for i, index in enumerate(df.index):
            new_index.append((i, *index[1:]))
        df.index = pd.MultiIndex.from_tuples(new_index, names=df.index.names)
        return df

    def run_all(self):
        self.minimize()

        # print("Running initial equilibration")
        # self.simulate(self.equil_steps, 1.0, 1.0, "equil")

        print("Removing electrostatics")
        lambda_ster = 1.0
        for lambda_elec in reversed(self.electrostatics_schedule):
            print(f"Running {lambda_ster=}, {lambda_elec=}")
            self.simulate(self.equil_steps, lambda_ster, lambda_elec)

        print("Removing sterics")
        lambda_elec = 0.0
        for lambda_ster in reversed(self.sterics_schedule):
            print(f"Running {lambda_ster=}, {lambda_elec=}")
            self.simulate(self.equil_steps, lambda_ster, lambda_elec)

        print("Re-adding electrostatics in vacuum")
        lambda_ster = 0.0
        lig_pos = self.system.get_lig_positions()
        self.system_vac.set_positions(lig_pos)
        for lambda_elec in self.electrostatics_schedule:
            print(f"Running {lambda_ster=}, {lambda_elec=}")
            self.simulate(self.equil_steps,
                          lambda_ster,
                          lambda_elec,
                          vacuum=True)

    def compute_delta_F(self):
        """ Compute the solvation free energy using MBAR """
        u_nk_vac = self.get_all_vac_u_nk()
        u_nk_solv = self.get_all_solv_u_nk()
        
        T = u_nk_vac.attrs["temperature"] * unit.kelvin

        mbar_vac = MBAR()
        mbar_vac.fit(u_nk_vac)

        mbar_solv = MBAR()
        mbar_solv.fit(u_nk_solv)

        from alchemlyb.visualisation import plot_mbar_overlap_matrix
        ax = plot_mbar_overlap_matrix(mbar_solv.overlap_matrix)
        ax.figure.savefig("/work/users/r/d/rdey/ml_implicit_solvent/mbar_overlap_matrix.png", dpi=300)
        
        F_solv_kt = -mbar_solv.delta_f_[(0, 0)][(1,1)] + mbar_vac.delta_f_[0][1]
        F_solv_dkt =  -mbar_solv.d_delta_f_[(0, 0)][(1,1)] + mbar_vac.d_delta_f_[0][1]
        F_solv = F_solv_kt * T * kB
        F_solv_error = F_solv_dkt * T * kB

        return F_solv.value_in_unit(unit.kilojoule_per_mole) * 0.239006, F_solv_error.value_in_unit(unit.kilojoule_per_mole) * 0.239006
        
