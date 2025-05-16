import copy
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import mdtraj as md
import openmm as mm
from openmm import app
from openmm import unit
from openmmtools.constants import kB
import alchemlyb
from alchemlyb.estimators import MBAR
from alchemlyb.preprocessing.subsampling import decorrelate_u_nk
from bigbind_solv.lr_complex import get_lr_complex
from bigbind_solv.fep import apply_fep, set_fep_lambdas
import pickle as pkl


def get_lig_and_water_indices(system):
    """Returns lists of sets of the ligand and water atom indices.
    This is the format needed by fep.py"""

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
    """Created a new alchemically modified LR system"""
    lig_indices, water_indices = get_lig_and_water_indices(system)

    # create a new alchemical system
    alchemical_system = apply_fep(system.system, lig_indices, water_indices)

    properties = {"Precision": "mixed", "DeviceIndex": "0"}

    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond
    )
    simulation = app.Simulation(
        system.topology, alchemical_system, integrator, system.platform, properties
    )

    lr_system = copy.copy(system)
    lr_system.system = alchemical_system
    lr_system.integrator = integrator
    lr_system.simulation = simulation

    return lr_system


class SolvationSim:
    """Class for running all the MD for the alchemical
    solvation free energy calculations."""

    def __init__(self, lig_file, out_folder, solvent):

        self.out_folder = out_folder
        os.makedirs(out_folder, exist_ok=True)

        is_explicit = solvent == "tip3p"

        solv_kwargs = {
            "nonbonded_cutoff": 0.9 * unit.nanometer,
            "constraints": app.HBonds,
            "box_padding": 1.6 * unit.nanometer,
            "lig_ff": "gaff",
            "cache_dir": out_folder,
            "solvent": solvent,
            "nonbonded_method": app.PME if is_explicit else app.NoCutoff,
            "include_barostat": is_explicit,
        }

        vac_kwargs = {
            "constraints": app.HBonds,
            "lig_ff": "gaff",
            "cache_dir": out_folder,
            "solvent": "none",
        }

        system = get_lr_complex(
            None,
            lig_file,
            **solv_kwargs,
        )
        system_vac = get_lr_complex(None, lig_file, **vac_kwargs)
        system.save(os.path.join(out_folder, "system"))
        system_vac.save(os.path.join(out_folder, "system_vac"))

        system.save_to_pdb(os.path.join(out_folder, "system.pdb"))
        system_vac.save_to_pdb(os.path.join(out_folder, "system_vac.pdb"))

        self.system = make_alchemical_system(system)
        self.system_vac = make_alchemical_system(system_vac)

        self.system.set_positions(system.get_positions())

        self.electrostatics_schedule = [
            0.00,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
        ]
        self.sterics_schedule = [
            0.00,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.00,
        ]

        self.equil_steps = 25000
        self.elapsed_time = 0  # will populate this after running the simulation

    def minimize(self):
        cache_file = os.path.join(self.out_folder, "minimized.pkl")
        tol = 1.0
        try:
            self.system.minimize_cached(cache_file, tol)
        except mm.OpenMMException:
            # in case we create a system with a different number of waters
            os.remove(cache_file)
            self.system.minimize_cached(cache_file, tol)

    def get_sim_prefix(self, lambda_sterics, lambda_electrostatics, vacuum=False):
        """Returns the prefix for the simulation file"""
        prefix = f"sim_{lambda_sterics}_{lambda_electrostatics}"
        if vacuum:
            prefix += "_vac"
        return prefix

    def simulate(
        self, n_steps, lambda_sterics, lambda_electrostatics, prefix=None, vacuum=False
    ):
        """Run the simulation for n_steps, saving the trajectory
        to out_folder/prefix.dcd. Returns the time taken to run the simulation"""
        if prefix is None:
            prefix = self.get_sim_prefix(lambda_sterics, lambda_electrostatics, vacuum)
        out_dcd = os.path.join(self.out_folder, f"{prefix}.dcd")
        out_com = os.path.join(self.out_folder, f"{prefix}.completed")
        cur_time = time.time()

        if os.path.exists(out_com):
            print(f"Skipping {prefix} -- already completed")
            elapsed_time = float(open(out_com).read())
            return elapsed_time

        if vacuum:
            simulation = self.system_vac.simulation
        else:
            simulation = self.system.simulation
        set_fep_lambdas(simulation.context, lambda_sterics, lambda_electrostatics)
        simulation.reporters.clear()
        simulation.reporters.append(app.DCDReporter(out_dcd, 1000))
        simulation.step(n_steps)

        elapsed_time = time.time() - cur_time

        with open(out_com, "w") as f:
            f.write(f"{elapsed_time}")

        return elapsed_time

    def get_vac_u_nk(self, lambda_elec, T=300 * unit.kelvin):
        """Returns the u_nk dataframe for the vacuum simulation
        run at lambda_elec. This returns the energy of the system
        from _all_ the lambda_elec values for the simulation"""

        prefix = self.get_sim_prefix(0.0, lambda_elec, vacuum=True)
        cache_fname = os.path.join(self.out_folder, f"{prefix}_u_nk.pkl")
        if os.path.exists(cache_fname):
            return pd.read_pickle(cache_fname)

        dcd_file = prefix + ".dcd"
        dcd_file = os.path.join(self.out_folder, dcd_file)
        pdb_file = os.path.join(self.out_folder, "system_vac.pdb")

        traj = md.load(dcd_file, top=pdb_file)
        df = pd.DataFrame(
            {
                "time": traj.time,
                "fep-lambda": [lambda_elec] * len(traj.time),
            }
        )
        df = df.set_index(["time", "fep-lambda"])

        for energy_lambda_elec in self.electrostatics_schedule:
            u = np.zeros(len(traj.time))
            set_fep_lambdas(self.system_vac.simulation.context, 0.0, energy_lambda_elec)
            for i, coords in enumerate(traj.xyz):
                self.system_vac.set_positions(coords * unit.nanometer)

                U = self.system_vac.simulation.context.getState(
                    getEnergy=True
                ).getPotentialEnergy()
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
        df = alchemlyb.concat(
            [
                self.get_vac_u_nk(lambda_elec)
                for lambda_elec in self.electrostatics_schedule
            ]
        )
        # make sure time is increasing
        new_index = []
        for i, index in enumerate(df.index):
            new_index.append((i, *index[1:]))
        df.index = pd.MultiIndex.from_tuples(new_index, names=df.index.names)
        return df

    def get_solv_lambda_schedule(self):
        """Returns a list of tuples of (lambda_ster, lambda_elec)
        for the solvation simulations"""

        lambda_schedule = []
        lambda_ster = 1.0
        for lambda_elec in reversed(self.electrostatics_schedule):
            lambda_schedule.append((lambda_ster, lambda_elec))

        lambda_elec = 0.0
        for lambda_ster in reversed(self.sterics_schedule):
            lambda_schedule.append((lambda_ster, lambda_elec))

        return lambda_schedule

    def get_solv_u_nk(
        self, lambda_ster, lambda_elec, T=300 * unit.kelvin, P=1.0 * unit.atmosphere
    ):
        """Returns the u_nk dataframe for the solvation simulation run
        at lambda_ster and lambda_elec."""

        prefix = self.get_sim_prefix(lambda_ster, lambda_elec)
        cache_fname = os.path.join(self.out_folder, f"{prefix}_u_nk.pkl")
        if os.path.exists(cache_fname):
            return pd.read_pickle(cache_fname)

        dcd_file = prefix + ".dcd"
        dcd_file = os.path.join(self.out_folder, dcd_file)
        pdb_file = os.path.join(self.out_folder, "system.pdb")

        traj = md.load(dcd_file, top=pdb_file)
        df = pd.DataFrame(
            {
                "time": traj.time,
                "vdw-lambda": [lambda_ster] * len(traj.time),
                "coul-lambda": [lambda_elec] * len(traj.time),
            }
        )
        df = df.set_index(["time", "vdw-lambda", "coul-lambda"])

        for energy_lambda_ster, energy_lambda_elec in self.get_solv_lambda_schedule():
            u = np.zeros(len(traj.time))
            set_fep_lambdas(
                self.system.simulation.context, energy_lambda_ster, energy_lambda_elec
            )
            if traj.unitcell_vectors is None:
                all_bvs = [None] * len(traj.xyz)
            else:
                all_bvs = [
                    box_vectors * unit.nanometer
                    for box_vectors in traj.unitcell_vectors
                ]
            for i, (coords, bvs) in enumerate(zip(traj.xyz, all_bvs)):
                self.system.set_positions(coords * unit.nanometer, box_vectors=bvs)
                U = self.system.simulation.context.getState(
                    getEnergy=True
                ).getPotentialEnergy()
                U_reduced = U / (kB * T)

                # add pV term if we have a box
                if bvs is not None:
                    beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * T)
                    V = bvs[0, 0] * bvs[1, 1] * bvs[2, 2]
                    U_reduced += P * V * beta

                # reduced energy (divided by kT)
                u[i] = U_reduced

            df[(energy_lambda_ster, energy_lambda_elec)] = u

        df.attrs = {
            "temperature": T.value_in_unit(unit.kelvin),
            "energy_unit": "kT",
        }

        df = decorrelate_u_nk(df, remove_burnin=True)

        df.to_pickle(cache_fname)

        return df

    def get_all_solv_u_nk(self):
        df = alchemlyb.concat(
            [
                self.get_solv_u_nk(lambda_ster, lambda_elec)
                for lambda_ster, lambda_elec in tqdm(self.get_solv_lambda_schedule())
            ]
        )
        # make sure time is increasing
        new_index = []
        for i, index in enumerate(df.index):
            new_index.append((i, *index[1:]))
        df.index = pd.MultiIndex.from_tuples(new_index, names=df.index.names)
        return df

    def run_all(self):
        self.minimize()

        print("Removing electrostatics")
        lambda_ster = 1.0
        for lambda_elec in reversed(self.electrostatics_schedule):
            print(f"Running {lambda_ster=}, {lambda_elec=}")
            self.elapsed_time += self.simulate(
                self.equil_steps, lambda_ster, lambda_elec
            )

        print("Removing sterics")
        lambda_elec = 0.0
        for lambda_ster in reversed(self.sterics_schedule):
            if lambda_ster == 1.0:
                continue

            print(f"Running {lambda_ster=}, {lambda_elec=}")
            self.elapsed_time += self.simulate(
                self.equil_steps, lambda_ster, lambda_elec
            )

        print("Re-adding electrostatics in vacuum")
        lambda_ster = 0.0
        lig_pos = self.system.get_lig_positions()
        self.system_vac.set_positions(lig_pos)
        for lambda_elec in self.electrostatics_schedule:
            print(f"Running {lambda_ster=}, {lambda_elec=}")
            self.elapsed_time += self.simulate(
                self.equil_steps, lambda_ster, lambda_elec, vacuum=True
            )

    def compute_delta_F(self):
        """Compute the solvation free energy using MBAR"""
        u_nk_vac = self.get_all_vac_u_nk()
        u_nk_solv = self.get_all_solv_u_nk()

        T = u_nk_vac.attrs["temperature"] * unit.kelvin

        mbar_vac = MBAR()
        mbar_vac.fit(u_nk_vac)

        mbar_solv = MBAR()
        mbar_solv.fit(u_nk_solv)

        F_solv_kt = mbar_vac.delta_f_[0][1] - mbar_solv.delta_f_[(0, 0)][(1, 1)]
        F_solv = F_solv_kt * T * kB

        self.vac_dF = (mbar_vac.delta_f_[0][1] * T * kB).value_in_unit(
            unit.kilocalories_per_mole
        )
        self.vac_ddF = (mbar_vac.d_delta_f_[0][1] * T * kB).value_in_unit(
            unit.kilocalories_per_mole
        )
        self.solv_dF = (mbar_solv.delta_f_[(0, 0)][(1, 1)] * T * kB).value_in_unit(
            unit.kilocalories_per_mole
        )
        self.solv_ddF = (mbar_solv.d_delta_f_[(0, 0)][(1, 1)] * T * kB).value_in_unit(
            unit.kilocalories_per_mole
        )

        ddF = self.vac_ddF + self.solv_ddF

        return F_solv.value_in_unit(unit.kilocalories_per_mole), ddF
