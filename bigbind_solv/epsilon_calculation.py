import openmm
from openmm.app.dcdreporter import DCDReporter
from openmm import app, unit
from create_dataset import SolvDatasetReporter
from fep import LAMBDA_STERICS, LAMBDA_ELECTROSTATICS, set_fep_lambdas
from scipy import stats
from lr_complex import LRComplex, get_lr_complex
import mdtraj as md
from freesolv import load_freesolv, smi_to_protonated_sdf
from sim import make_alchemical_system
import numpy as np
import os
import sys
import glob
import random
import shutil
import itertools
from tqdm import tqdm
import h5py
from config_dict import config_dict


#This was basically copy pasted from the original SolvReporter
# with some modifications for dp management
class SolvDatasetReporterWithCustomDP(SolvDatasetReporter):
    """ 
    Subclass of SolvDatasetReporter that allows custom dp for finite difference in derivatives.
    """

    def __init__(self, filename, system_vac, report_interval, dp=1e-5):
        super().__init__(filename, system_vac, report_interval)

        self.dp = dp

    def get_parameter_derivative(self, simulation, param_name, dp=None):
        """
        Uses finite difference to calculate the derivative of the 
        potential energy with respect to a parameter, with a customizable dp value.
        """
        dp = dp if dp is not None else self.dp

        context = simulation.context
        parameter = context.getParameter(param_name)

        if np.isclose(parameter, 1.0, 1e-4):
            dp = -dp

        initial_energy = context.getState(getEnergy=True).getPotentialEnergy()

        context.setParameter(param_name, parameter + dp)
        final_energy = context.getState(getEnergy=True).getPotentialEnergy()

        context.setParameter(param_name, parameter)

        return (final_energy - initial_energy) / dp

    def report(self, simulation, state):

        positions = state.getPositions(asNumpy=True).value_in_unit(
            unit.nanometer)
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer)
        U = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        sterics_derivative = self.get_parameter_derivative(
            simulation, LAMBDA_STERICS)
        electrostatics_derivative = self.get_parameter_derivative(
            simulation, LAMBDA_ELECTROSTATICS)

        self.system_vac.set_positions(positions[self.mol_indices] *
                                      unit.nanometer)
        vac_forces = self.system_vac.get_forces().value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer)

        vac_electrostatics_derivative = self.get_parameter_derivative(
            self.system_vac.simulation, LAMBDA_ELECTROSTATICS)
        lambda_sterics = simulation.context.getParameter(LAMBDA_STERICS)
        lambda_electrostatics = simulation.context.getParameter(
            LAMBDA_ELECTROSTATICS)

        self.file["positions"].resize(
            (self.file["positions"].shape[0] + 1, len(self.mol_indices), 3))
        self.file["positions"][-1] = positions[self.mol_indices]

        self.file["solv_forces"].resize(
            (self.file["solv_forces"].shape[0] + 1, len(self.mol_indices), 3))
        self.file["solv_forces"][-1] = forces[self.mol_indices] - vac_forces

        self.file["sterics_derivatives"].resize(
            (self.file["sterics_derivatives"].shape[0] + 1, ))
        self.file["sterics_derivatives"][
            -1] = sterics_derivative.value_in_unit(unit.kilojoules_per_mole)

        self.file["electrostatics_derivatives"].resize(
            (self.file["electrostatics_derivatives"].shape[0] + 1, ))
        self.file["electrostatics_derivatives"][-1] = (
            electrostatics_derivative -
            vac_electrostatics_derivative).value_in_unit(
                unit.kilojoules_per_mole)

        self.file["energies"].resize((self.file["energies"].shape[0] + 1, ))
        self.file["energies"][-1] = U

        self.file["lambda_sterics"].resize(
            (self.file["lambda_sterics"].shape[0] + 1, ))
        self.file["lambda_sterics"][-1] = lambda_sterics

        self.file["lambda_electrostatics"].resize(
            (self.file["lambda_electrostatics"].shape[0] + 1, ))
        self.file["lambda_electrostatics"][-1] = lambda_electrostatics


def runSim(base_file_path, sim_file_path, steps, dp, lambda_electrostatics,
           lambda_sterics):

    out_file = os.path.join(sim_file_path, "sim.h5")
    if os.path.exists(out_file):
        os.remove(out_file)

    #load all cached files that can be reused
    try:
        system_xml = os.path.join(base_file_path, "system")
        system_pkl = os.path.join(base_file_path, "system")
        system_vac_xml = os.path.join(base_file_path, "system_vac")
        system_vac_pkl = os.path.join(base_file_path, "system_vac")
        system = LRComplex.load(system_xml, system_pkl)
        system_vac = LRComplex.load(system_vac_xml, system_vac_pkl)
    except Exception as e:
        print(f"Files not found for path: {base_file_path}... continuing...")
        return

    vac_indicies = system.lig_indices
    #freeze ligand
    for i in vac_indicies:
        system.system.setParticleMass(i, 0.0)

    alc_system = make_alchemical_system(system)
    alc_system_vac = make_alchemical_system(system_vac)

    alc_system.set_positions(system.get_positions())

    set_fep_lambdas(alc_system.simulation.context, lambda_sterics,
                    lambda_electrostatics)
    set_fep_lambdas(alc_system_vac.simulation.context, lambda_sterics,
                    lambda_electrostatics)

    simulation = alc_system.simulation
    simulation.reporters = []

    simulation.minimizeEnergy()
    reporter = SolvDatasetReporterWithCustomDP(out_file, alc_system_vac, 500,
                                               dp)
    simulation.reporters.append(reporter)
    simulation.reporters.append(
        DCDReporter(os.path.join(sim_file_path, "simulation.dcd"), 100))

    simulation.step(steps)


def computationalStability(file_path, folder_paths):
    lambda_elecs = [0.99999973, 1]
    lambda_sters = [0.99999973, 1]
    steps = 250000
    dp = 1e-4
    com = list(itertools.product(lambda_elecs, lambda_sters))

    for (lambda_elec, lambda_ster) in com:
        for folder in tqdm(folder_paths):
            try:
                folder = os.path.join(file_path, folder)
                assert os.path.exists(folder)
                sim_path = f"Stability_check_{lambda_elec}_{lambda_ster}"
                sim_folder = os.path.join(folder, sim_path)
                if os.path.exists(sim_folder):
                    shutil.rmtree(sim_folder)
                os.mkdir(sim_folder)
                runSim(folder, sim_folder, steps, dp, lambda_elec, lambda_ster)
            except Exception as e:
                print(f"Failed to run: {str(e)}... Continuing...")


def runAll(file_path, folder_paths, sim_path, dp, lambda_elec):
    lambda_sterics = 0.99999973
    steps = 250000
    for folder in tqdm(folder_paths):
        try:
            folder = os.path.join(file_path, folder)
            assert os.path.exists(folder)
            sim_folder = os.path.join(folder, sim_path)
            if os.path.exists(sim_folder):
                shutil.rmtree(sim_folder)
            os.mkdir(sim_folder)
            runSim(folder, sim_folder, steps, dp, lambda_elec, lambda_sterics)
        except Exception as e:
            print("Failed to run... Continuing... ")


#maintain same folders for calculations
def randomSample(bigbind_path):
    if os.path.exists(bigbind_path):
        folders = [
            f for f in glob.glob(os.path.join(bigbind_path, "[0-9]*"))
            if os.path.isdir(f)
        ]
        random_folders = random.sample(folders, 500)
        random_folders = random.sample([os.path.basename(f) for f in folders],
                                       500)

        with open(
                os.path.join(config_dict['base_file'],
                             "bigbind_solv/Random_files.txt"), "w") as f:
            for folder in random_folders:
                f.write(f"{folder}\n")
        return random_folders


#=========MAIN FUNCTIONS=========


#analyze existing stats
def statsAnalysis(base_path, folder_paths):
    dps = [1e-4]
    lambda_elecs = [1]
    com = list(itertools.product(lambda_elecs, dps))

    for (lambda_elec, dp) in com:
        array = []
        for folder in folder_paths:

            sim_path = os.path.join(base_path, folder,
                                    f"Epsilon_{dp}_{lambda_elec}", "sim1M.h5")
            print(sim_path)
            if not os.path.exists(sim_path):
                continue
            try:
                file = h5py.File(sim_path, 'r')
                mean_elec = np.mean(file['electrostatics_derivatives'])
                mean_U = np.mean(file['energies'])
                array.append(mean_elec)
                file.close()
            except OSError:
                print(f"Skipping corrupt or unreadable file: {sim_path}")
                continue
        array = np.array(array)
        change = np.nanstd(array)
        avg = np.nanmean(array)
        print(
            f"Current DP: {dp}; Current Lambda_Elec {lambda_elec} STD: {change}; Average {avg}"
        )

    return


def StabilitystatsAnalysis(base_path, folder_paths):
    lambda_elecs = [0.99999973, 1]
    lambda_sters = [0.99999973, 1]
    com = list(itertools.product(lambda_elecs, lambda_sters))

    for (lambda_elec, lambda_ster) in com:
        array = []
        for folder in folder_paths:
            sim_path = os.path.join(
                base_path, folder,
                f"Stability_check_{lambda_elec}_{lambda_ster}", "sim.h5")
            if not os.path.exists(sim_path):
                continue
            try:
                file = h5py.File(sim_path, 'r')

                if (len(file['electrostatics_derivatives']) < 200):
                    continue
                mean_elec = np.nanmean(file['sterics_derivatives'])
                mean_U = np.nanmean(file['energies'])

                array.append(
                    [mean_elec,
                     np.array(file['sterics_derivatives'])])
            except Exception as e:
                print(
                    f"Skipping corrupt or unreadable file: {sim_path}: {str(e)}"
                )
                continue
        #array = np.array(array)
        #array = np.array(array[:, 0])
        #change = np.std(array[:, 0])
        min_val = max(
            array,
            key=lambda x: x[0])  # Find the row with the smallest mean elec
        print(lambda_elec, lambda_ster, max(min_val[1]))
        #avg = np.nanmean(array[:, 0])
        #print(f"Current Lambda_Elec {lambda_elec} Lambda_Ster {lambda_ster} PTP: {change}; Average {avg}")


def simulate(file_path):

    dp = float(sys.argv[1])
    assert dp in [1e-3, 1e-2, 1e-4]

    lambda_elec = int(sys.argv[2])
    assert lambda_elec in [0, 1]

    new_sim_path_name = f"Epsilon_{dp}_{lambda_elec}"

    with open(
            "/work/users/r/d/rdey/ml_implicit_solvent/bigbind_solv/Random_files.txt",
            "r") as file:
        folder_array = [line.strip() for line in file]

    runAll(file_path, folder_array, new_sim_path_name, dp, lambda_elec)


if __name__ == "__main__":
    file_path = config_dict['bind_dir']
    with open(
            "/work/users/r/d/rdey/ml_implicit_solvent/bigbind_solv/Random_files.txt",
            "r") as file:
        folder_array = [line.strip() for line in file]
    StabilitystatsAnalysis(file_path, folder_array)
