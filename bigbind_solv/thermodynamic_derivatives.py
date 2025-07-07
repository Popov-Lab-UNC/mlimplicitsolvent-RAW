from openmmtools.alchemy import AbsoluteAlchemicalFactory
from openmmtools import alchemy
from openmmtools.states import ThermodynamicState, CompoundThermodynamicState, SamplerState
from openmm import unit, LangevinMiddleIntegrator, Platform, app
from openmm.app.dcdreporter import DCDReporter
import openmm
from openmm.app import *
from create_dataset import SolvDatasetReporter
from lr_complex import LRComplex, get_lr_complex
import os
import alchemlyb
from alchemlyb.estimators import MBAR
from alchemlyb.preprocessing.subsampling import decorrelate_u_nk
from openmmtools.constants import kB
import numpy as np
import h5py
import subprocess
import time
from tqdm import tqdm
import pandas as pd
from openmm import LocalEnergyMinimizer

#Not Needed for Solvation Calculations-- Used for dataset creation.
class ThermodynamicDerivativesReporter(SolvDatasetReporter):
    def get_parameter_derivative(context, param_name, dp=1e-4):
        """ 
        Uses finite difference to calculate the derivative of the 
        potential energy with respect to a parameter.
        """
        parameter = context.getParameter(param_name)

        if (parameter == 1.0):
            dp = -dp

        initial_energy = context.getState(getEnergy=True).getPotentialEnergy()

        context.setParameter(param_name, parameter + dp)
        final_energy = context.getState(getEnergy=True).getPotentialEnergy()

        context.setParameter(param_name, parameter)

        return (final_energy - initial_energy) / dp

    """ 
    Subclass of SolvDatasetReporter that allows custom dp for finite difference in derivatives.
    """

    def __init__(self, filename, system, system_vac_context, report_interval):
        super().__init__(filename, system, report_interval)
        self.system_vac_context = system_vac_context

    def report(self, context, state):

        positions = state.getPositions(asNumpy=True).value_in_unit(
            unit.nanometer)
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer)
        U = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        derivatives = state.getEnergyParameterDerivatives()

        print("Available derivative keys:", list(derivatives.values()))

        sterics_derivative = get_parameter_derivatives(context,
                                                       "lambda_sterics")
        electrostatics_derivative = get_parameter_derivatives(
            context, 'lambda_electrostatics')

        print("sterics_derivative:", sterics_derivative)
        print("electrostatics_derivative:", electrostatics_derivative)

        self.system_vac_context.setPositions(positions[self.mol_indices] *
                                             unit.nanometer)
        state_vac = self.system_vac_context.getState(
            getForces=True, getParameterDerivatives=True)

        vac_forces = state_vac.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer)

        print("vac_forces:", vac_forces)

        vac_electrostatics_derivative = get_parameter_derivatives(
            self.system_vac_context, "lambda_electrostatics")

        self.file["positions"].resize(
            (self.file["positions"].shape[0] + 1, len(self.mol_indices), 3))
        self.file["positions"][-1] = positions[self.mol_indices]

        self.file["solv_forces"].resize(
            (self.file["solv_forces"].shape[0] + 1, len(self.mol_indices), 3))
        self.file["solv_forces"][-1] = forces[self.mol_indices] - vac_forces

        self.file["sterics_derivatives"].resize(
            (self.file["sterics_derivatives"].shape[0] + 1, ))
        self.file["sterics_derivatives"][-1] = sterics_derivative

        self.file["electrostatics_derivatives"].resize(
            (self.file["electrostatics_derivatives"].shape[0] + 1, ))
        #self.file["electrostatics_derivatives"][-1] = electrostatics_derivative.value_in_unit(unit.kilojoules_per_mole)
        self.file["electrostatics_derivatives"][-1] = (
            electrostatics_derivative - vac_electrostatics_derivative)

        self.file["energies"].resize((self.file["energies"].shape[0] + 1, ))
        self.file["energies"][-1] = U

        self.file["lambda_sterics"].resize(
            (self.file["lambda_sterics"].shape[0] + 1, ))
        self.file["lambda_sterics"][-1] = lambda_sterics

        self.file["lambda_electrostatics"].resize(
            (self.file["lambda_electrostatics"].shape[0] + 1, ))
        self.file["lambda_electrostatics"][-1] = lambda_electrostatics


#only need positions from DCD, but cant use dcd so impromptu function:
class PositionsReporter:
    """Saves ligand-only positions to an HDF5 file at specified intervals."""
    def __init__(self, filename, mol_indices, report_interval):
        self.report_interval = report_interval
        self.mol_indices = mol_indices

        self.file = h5py.File(filename, 'w')
        self.file.create_dataset("positions",
                                 maxshape=(None, len(self.mol_indices), 3),
                                 shape=(0, len(self.mol_indices), 3),
                                 dtype=np.float32)

    def __del__(self):
        self.file.close()

    def describeNextReport(self, simulation):
        steps = self.report_interval - simulation.currentStep % self.report_interval
        return (steps, True, False, False, False, None)

    def report(self, simulation, state):
        positions = state.getPositions(asNumpy=True).value_in_unit(
            unit.nanometer)

        # Resize and append ligand positions
        current_length = self.file["positions"].shape[0]
        self.file["positions"].resize(
            (current_length + 1, len(self.mol_indices), 3))
        self.file["positions"][-1] = positions[self.mol_indices]



#Primary Class for Implicit Solvation Calculation
class ImplicitSolv:
    @staticmethod
    def create_system(out_folder, lig_file, smile, solvent):
        kwargs = {
            "nonbonded_cutoff": 0.9 * unit.nanometer,
            # "nonbonded_cutoff": 1.5*unit.nanometer,
            "constraints": app.HBonds,
            "box_padding": 1.6 * unit.nanometer,
            # "box_padding": 2.0*unit.nanometer,
            "lig_ff": "gaff",
            "cache_dir": out_folder,
        }
        lig_file = os.path.join(out_folder, "ligand.sdf")
        if not os.path.exists(lig_file):
            cmd = f'obabel "-:{smile}" -O "{lig_file}" --gen3d --pH 7'
            subprocess.run(cmd, check=True, shell=True, timeout=60)
        system = get_lr_complex(
            None,
            lig_file,
            solvent=solvent,
            include_barostat=True if solvent == "tip3p" else False,
            **kwargs)
        system_vac = get_lr_complex(None, lig_file, solvent="none", **kwargs)

        system.save(os.path.join(out_folder, "system"))
        system_vac.save(os.path.join(out_folder, "system_vac"))

        system.save_to_pdb(os.path.join(out_folder, "system.pdb"))
        system_vac.save_to_pdb(os.path.join(out_folder, "system_vac.pdb"))
        return system, system_vac

    def __init__(self, base_path, name, smile, solvent):
        self._T = 300 * unit.kelvin
        self.platform = Platform.getPlatformByName("CUDA")
        self.electrostatics = [0.0, 0.5, 1.0]
        self.sterics = [
            0.0, 1.0
        ]
        self.name = name
        self.n_steps = 10000
        self.report_interval = 1000
        self.compound_state, self.compound_state_vac, self.system, self.system_vac = self.createCompoundStates(
            base_path, smile, name, solvent)

    def get_solv_lambda_schedule(self):
        """ Returns a list of tuples of (lambda_ster, lambda_elec) 
        for the solvation simulations """

        lambda_schedule = []
        lambda_ster = 1.0
        for lambda_elec in reversed(self.electrostatics):
            lambda_schedule.append((lambda_ster, lambda_elec))

        lambda_elec = 0.0
        for lambda_ster in reversed(self.sterics):
            lambda_schedule.append((lambda_ster, lambda_elec))

        return lambda_schedule

    def createCompoundStates(self, base_path, smile, name, solvent):
        self.name_path = os.path.join(base_path, name)
        system_path = os.path.join(self.name_path, "system")
        lig_path = os.path.join(self.name_path, "ligand.sdf")

        if (not os.path.exists(self.name_path)):
            os.mkdir(self.name_path)
        if (not os.path.exists(system_path)):
            system, system_vac = self.create_system(self.name_path, lig_path,
                                                    smile, solvent)

        else:
            try:
                system_vac_path = os.path.join(self.name_path, "system_vac")
                system = LRComplex.load(system_path)
                system_vac = LRComplex.load(system_vac_path)
            except Exception as e:
                print(f"Existing files corrupted... Continuing...: {str(e)}")
                return

        self.PDB = PDBFile(os.path.join(self.name_path, "system.pdb"))
        self.mol_indices = system_vac.lig_indices

        factory = AbsoluteAlchemicalFactory(
            consistent_exceptions=False,
            alchemical_pme_treatment='direct-space',
            disable_alchemical_dispersion_correction=True,
            split_alchemical_forces=True)

        system_region = alchemy.AlchemicalRegion(
            alchemical_atoms=system.lig_indices,
            annihilate_electrostatics=True,
            annihilate_sterics=False)
        system_vac_region = alchemy.AlchemicalRegion(
            alchemical_atoms=system_vac.lig_indices,
            annihilate_electrostatics=True,
            annihilate_sterics=False)

        print("Alchemical Atoms:", system_region.alchemical_atoms)

        alchemy_system = factory.create_alchemical_system(
            system.system, system_region)
        alchemy_system_vac = factory.create_alchemical_system(
            system_vac.system, system_vac_region)

        therm_state = ThermodynamicState(alchemy_system, self._T)
        therm_vac_state = ThermodynamicState(alchemy_system_vac, self._T)

        alchemical_state = alchemy.AlchemicalState.from_system(alchemy_system)

        alchemical_state_vac = alchemy.AlchemicalState.from_system(
            alchemy_system_vac)

        compound_state = CompoundThermodynamicState(
            thermodynamic_state=therm_state,
            composable_states=[alchemical_state])
        compound_state_vac = CompoundThermodynamicState(
            thermodynamic_state=therm_vac_state,
            composable_states=[alchemical_state_vac])

        return compound_state, compound_state_vac, alchemy_system, alchemy_system_vac

    def run_sim(self, lambda_elec, lambda_ster, vacuum=False):
        if vacuum:
            compound_state = self.compound_state_vac
            file_name = os.path.join(
                self.name_path, f"{self.name}_vac_{lambda_elec}_{lambda_ster}")
        else:
            compound_state = self.compound_state
            file_name = os.path.join(
                self.name_path, f"{self.name}_{lambda_elec}_{lambda_ster}")

        compound_state.lambda_electrostatics = lambda_elec
        compound_state.lambda_sterics = lambda_ster

        com_file_name = file_name + ".com"
        file_name += ".h5"

        if (os.path.exists(com_file_name)):
            print("File Already Exists, Continuing...")
            return

        integrator = LangevinMiddleIntegrator(self._T, 1.0 / unit.picoseconds,
                                              0.002 * unit.picoseconds)

        self.curr_context = compound_state.create_context(
            integrator, self.platform)
        print(self.PDB.positions)
        self.curr_context.setPositions(self.PDB.positions)
        LocalEnergyMinimizer.minimize(self.curr_context)
        reporter_solv = PositionsReporter(file_name, self.mol_indices,
                                          self.report_interval)

        for _ in range(0, self.n_steps, self.report_interval):
            integrator.step(self.report_interval)
            state = self.curr_context.getState(getPositions=True,
                                               getForces=True,
                                               getEnergy=True,
                                               getParameters=True,
                                               getParameterDerivatives=True)
            #reporter_dcd.report(context, state)
            reporter_solv.report(self.curr_context, state)

        with open(com_file_name, 'w'):
            pass

    def run_all(self):
        print("Removing electrostatics")
        lambda_ster = 1.0
        for lambda_elec in reversed(self.electrostatics):
            print(f"Running {lambda_ster=}, {lambda_elec=}")
            self.run_sim(lambda_elec, lambda_ster)

        print("Removing sterics")
        lambda_elec = 0.0
        for lambda_ster in reversed(self.sterics):
            print(f"Running {lambda_ster=}, {lambda_elec=}")
            self.run_sim(lambda_elec, lambda_ster)

        print("Re-adding electrostatics in vacuum")
        lambda_ster = 0.0

        for lambda_elec in self.electrostatics:
            print(f"Running {lambda_ster=}, {lambda_elec=}")
            self.run_sim(lambda_elec, lambda_ster, vacuum=True)

    def calculate_energy_for_traj(self, pos, energy_context, e_lambda_ster, e_lambda_elec, vacuum):
        u = np.zeros(len(pos))
        if vacuum:
            compound_state = self.compound_state_vac
        else:
            compound_state = self.compound_state

        compound_state.lambda_electrostatics = e_lambda_elec
        compound_state.lambda_sterics = e_lambda_ster
        compound_state.apply_to_context(energy_context)

        for idx, coords in enumerate(pos):
            energy_context.setPositions(coords)
            #LocalEnergyMinimizer.minimize(energy_context)
            U = energy_context.getState(getEnergy=True).getPotentialEnergy()

            u[idx] = U / (kB * self._T)
        return u
    

    def get_vac_u_nk(self):
        cache_path = os.path.join(self.name_path, f"{self.name}_vac_u_nk.pkl")
        '''
        if os.path.exists(cache_path):
            return pd.read_pickle(cache_path)
        '''
        
        integrator = LangevinMiddleIntegrator(self._T, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
        energy_context = self.compound_state_vac.create_context(integrator, self.platform)

        vac_u_nk_df = []

        for lambda_elec in self.electrostatics:
            indiv_path = os.path.join(
                self.name_path,
                f"{self.name}_{lambda_elec}_vac_u_nk.pkl")
            if os.path.exists(indiv_path):
                df = pd.read_pickle(indiv_path)
                df = self.u_nk_processing_df(df)
                vac_u_nk_df.append(df)
            file_name = os.path.join(
            self.name_path, f"{self.name}_vac_{lambda_elec}_0.0")
            file_name += ".h5"

            with h5py.File(file_name, 'r') as f:
                pos = f['positions'][:]
            df = pd.DataFrame({
                "time": [
                    self.report_interval * idx * 0.002
                    for idx, _ in enumerate(pos)
                ],
                "fep-lambda": [lambda_elec] * len(pos),
            })
            df = df.set_index(["time", "fep-lambda"])

            for e_lambda_elec in self.electrostatics:
                u = self.calculate_energy_for_traj(pos,energy_context, 0.0,
                                                   e_lambda_elec, True)
                df[(e_lambda_elec)] = u

            #df.to_pickle(indiv_path)
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





            
            


    def get_solv_u_nk(self):

        cache_path = os.path.join(self.name_path, f"{self.name}_u_nk.pkl")

        if os.path.exists(cache_path):
            return pd.read_pickle(cache_path)
        
        integrator = LangevinMiddleIntegrator(self._T, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
        energy_context = self.compound_state.create_context(integrator, self.platform)
            

        solv_u_nk_df = []

        for (lambda_ster,
             lambda_elec) in tqdm(self.get_solv_lambda_schedule()):
            ''
            indiv_path = os.path.join(
                self.name_path,
                f"{self.name}_{lambda_elec}_{lambda_ster}_u_nk.pkl")
            if os.path.exists(indiv_path):
                df = pd.read_pickle(indiv_path)
                df = self.u_nk_processing_df(df)
                solv_u_nk_df.append(df)
                continue
            file_name = os.path.join(
                self.name_path, f"{self.name}_{lambda_elec}_{lambda_ster}")
            file_name += ".h5"
            with h5py.File(file_name, 'r') as f:
                pos = f['positions'][:]
            df = pd.DataFrame({
                "time": [
                    self.report_interval * idx * 0.002
                    for idx, _ in enumerate(pos)
                ],
                "vdw-lambda": [lambda_ster] * len(pos),
                "coul-lambda": [lambda_elec] * len(pos),
            })
            df = df.set_index(["time", "vdw-lambda", "coul-lambda"])

            for (e_lambda_ster,
                 e_lambda_elec) in self.get_solv_lambda_schedule():
                u = self.calculate_energy_for_traj(pos,energy_context, e_lambda_ster,
                                                   e_lambda_elec, False)
                df[(e_lambda_ster, e_lambda_elec)] = u

            #df.to_pickle(indiv_path)
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

    def u_nk_processing_df(self, df):
        df.attrs = {
            "temperature": self._T,
            "energy_unit": "kT",
        }

        df = alchemlyb.preprocessing.decorrelate_u_nk(df, remove_burnin=True)
        return df

    def compute_delta_F(self):
        """ Compute the solvation free energy using MBAR """
        u_nk_vac = self.get_vac_u_nk()
        u_nk_solv = self.get_solv_u_nk()

        mbar_vac = MBAR()
        mbar_vac.fit(u_nk_vac)

        mbar_solv = MBAR()
        mbar_solv.fit(u_nk_solv)

        from alchemlyb.visualisation import plot_mbar_overlap_matrix
        ax = plot_mbar_overlap_matrix(mbar_solv.overlap_matrix)
        ax.figure.savefig(
            "/work/users/r/d/rdey/ml_implicit_solvent/mbar_overlap_matrix.png",
            dpi=300)

        F_solv_kt = -mbar_solv.delta_f_[
            (0, 0)][(1, 1)] + mbar_vac.delta_f_[0][1]
        F_solv_dkt = -mbar_solv.d_delta_f_[
            (0, 0)][(1, 1)] + mbar_vac.d_delta_f_[0][1]
        F_solv = F_solv_kt * kB * self._T
        F_solv_error = F_solv_dkt * kB * self._T

        return F_solv.value_in_unit(
            unit.kilojoule_per_mole) * 0.239006, F_solv_error.value_in_unit(
                unit.kilojoule_per_mole) * 0.239006

import sys
if __name__ == "__main__":


    base_file_path = "/work/users/r/d/rdey/FINAL_OBC2"
    smile = str(sys.argv[1])
    expt = float(sys.argv[2])
    name = str(sys.argv[3])
    print(f"Current: {name}, {smile}, {expt}")
    start = time.time()
    main = ImplicitSolv(base_file_path, name,
                        smile, "obc2")
    main.run_all()
    res = main.compute_delta_F()
    print(time.time() - start)
    print(f"{name}, {res}, {expt}")

        
