from config_dict import config_dict
import openmm as mm
from openmm import unit, app
from openmm.app.dcdreporter import DCDReporter
from sim import get_lig_and_water_indices
import copy
from epsilon_calculation import SolvDatasetReporterWithCustomDP, runSim
from fep import apply_fep, LAMBDA_STERICS, LAMBDA_ELECTROSTATICS, set_fep_lambdas
import os
from lr_complex import LRComplex, get_lr_complex


#taken from sim.py and changed from stepsize and integrator debugging
def make_modified_alchemical_system(system):
    """ Created a new alchemically modified LR system"""
    lig_indices, water_indices = get_lig_and_water_indices(system)

    # create a new alchemical system
    alchemical_system = apply_fep(system.system, lig_indices, water_indices)

    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin,
                                             1.0 / unit.picosecond,
                                             1.0 * unit.femtosecond)
    simulation = app.Simulation(system.topology, alchemical_system, integrator,
                                system.platform)

    lr_system = copy.copy(system)
    lr_system.system = alchemical_system
    lr_system.integrator = integrator
    lr_system.simulation = simulation

    return lr_system


#taken from epsilon_calculation and changed from modified system
def runModSim(base_file_path, sim_file_path, steps, dp, lambda_electrostatics,
              lambda_sterics):

    out_file = os.path.join(sim_file_path, "sim.h5")
    simulation_file = os.path.join(sim_file_path, "simulation.dcd")
    if os.path.exists(out_file):
        os.remove(out_file)
        os.remove(simulation_file)

    #load all cached files that can be reused

    system_file = os.path.join(base_file_path, "system")

    system_vac_file = os.path.join(base_file_path, "system_vac")
    system = LRComplex.load(system_file, cuda=False)
    system_vac = LRComplex.load(system_vac_file, cuda=False)

    vac_indicies = system.lig_indices
    #freeze ligand
    for i in vac_indicies:
        system.system.setParticleMass(i, 0.0)

    alc_system = make_modified_alchemical_system(system)
    alc_system_vac = make_modified_alchemical_system(system_vac)

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
    simulation.reporters.append(DCDReporter(simulation_file, 100))

    simulation.step(steps)


if __name__ == "__main__":
    file = os.path.join(config_dict["bind_dir"], str(112014))
    sim = "Integrator"
    non_sim = "Non_Integrator"
    sim_path = os.path.join(file, sim)
    non_sim_path = os.path.join(file, non_sim)
    if (not os.path.exists(sim_path)):
        os.mkdir(sim_path)
    if (not os.path.exists(non_sim_path)):
        os.mkdir(non_sim_path)
    print("Running Middle Integrator with 1fs")
    runModSim(file, sim_path, 100000, 1e-5, 1, 1)
    print("Running Integrator with 2fs")
    runSim(file, non_sim_path, 100000, 1e-5, 1, 1)
