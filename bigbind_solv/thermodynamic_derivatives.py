from openmmtools.alchemy import AbsoluteAlchemicalFactory
from openmmtools import alchemy
from openmmtools.states import ThermodynamicState, CompoundThermodynamicState
from openmm import unit, LangevinMiddleIntegrator, Platform, app
from openmm.app.dcdreporter import DCDReporter
import openmm
from create_dataset import SolvDatasetReporter
from lr_complex import LRComplex, get_lr_complex
import os


class ThermodynamicDervivativesReporter(SolvDatasetReporter):
    """ 
    Subclass of SolvDatasetReporter that allows custom dp for finite difference in derivatives.
    """

    def __init__(self, filename, system_vac, system_vac_simulation, report_interval):
        super().__init__(filename, system_vac, report_interval)
        self.system_vac_simulation = system_vac_simulation

    def report(self, simulation, state):

        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)
        U = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        derivatives = state.getEnergyParameterDerivatives()


        sterics_derivative = derivatives["lambda_sterics"].value_in_unit(unit.kilojoules_per_mole)
        electrostatics_derivative = derivatives["lambda_electrostatics"].value_in_unit(unit.kilojoules_per_mole)

        context_vac = self.system_vac_simulation.context
        context_vac.setPositions(positions[self.mol_indices] * unit.nanometer)
        state_vac = context_vac.getState(getForces=True)

        vac_forces = state_vac.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)

        #dont need for decoupling
        #vac_electrostatics_derivative = state_vac.getEnergyParameterDerivatives()["lambda_electrostatics"].value_in_unit(unit.kilojoules_per_mole)

        lambda_sterics = simulation.context.getParameter('lambda_sterics')
        lambda_electrostatics = simulation.context.getParameter('lambda_electrostatics')

        self.file["positions"].resize((self.file["positions"].shape[0] + 1, len(self.mol_indices), 3))
        self.file["positions"][-1] = positions[self.mol_indices]

        self.file["solv_forces"].resize((self.file["solv_forces"].shape[0] + 1, len(self.mol_indices), 3))
        self.file["solv_forces"][-1] = forces[self.mol_indices] - vac_forces

        self.file["sterics_derivatives"].resize((self.file["sterics_derivatives"].shape[0] + 1,))
        self.file["sterics_derivatives"][-1] = sterics_derivative.value_in_unit(unit.kilojoules_per_mole)

        self.file["electrostatics_derivatives"].resize((self.file["electrostatics_derivatives"].shape[0] + 1,))
        self.file["electrostatics_derivatives"][-1] = electrostatics_derivative.value_in_unit(unit.kilojoules_per_mole)
        #self.file["electrostatics_derivatives"][-1] = (electrostatics_derivative - vac_electrostatics_derivative).value_in_unit(unit.kilojoules_per_mole)

        self.file["energies"].resize((self.file["energies"].shape[0] + 1,))
        self.file["energies"][-1] = U

        self.file["lambda_sterics"].resize((self.file["lambda_sterics"].shape[0] + 1,))
        self.file["lambda_sterics"][-1] = lambda_sterics

        self.file["lambda_electrostatics"].resize((self.file["lambda_electrostatics"].shape[0] + 1,))
        self.file["lambda_electrostatics"][-1] = lambda_electrostatics


#will do later lol
def create_system():
    return

def freeze_atoms(system, lig_indicies):
    for i in lig_indicies:
        system.system.setParticleMass(i, 0.0)
    return system

def runSim(base_path, lig_index, sim_path, lambda_elec, lambda_sterics, steps, stepSize):

    T = 300*unit.kelvin
    path = os.path.join(base_path, lig_index)
    dcd_path = os.path.join(path, sim_path, 'simulation.dcd')
    data_path = os.path.join(path, sim_path, 'sim.h5')

    if(not os.path.exists(path)):
        create_system()
    else:
        try:
            system_path = os.path.join(path, "system")
            system_vac_path = os.path.join(path, "system_vac")
            system = LRComplex.load(system_path)
            system_vac = LRComplex.load(system_vac_path)
        except:
            print("Existing files corrupted... Continuing...")
            return


    

    factory = AbsoluteAlchemicalFactory(
        consistent_exceptions=True,
        alchemical_pme_treatment='exact'
    )   

    #Change to annihilate = True if improvement is not significant
    system_region = alchemy.AlchemicalRegion(alchemical_atoms = system.lig_indices, annihilate_electrostatics = False, annihilate_sterics = False)
    system_vac_region = alchemy.AlchemicalRegion(alchemical_atoms = system_vac.lig_indices, annihilate_electrostatics = False, annihilate_sterics = False)

    alchemy_system = factory.create_alchemical_system(system.system, system_region)
    alchemy_system_vac = factory.create_alchemical_system(system_vac.system, system_vac_region)

    therm_state = ThermodynamicState(system.system, T)
    therm_vac_state = ThermodynamicState(system_vac.system, T)
    
    alchemical_state = alchemy.AlchemicalState.from_system(alchemy_system)
    alchemical_state_vac = alchemy.AlchemicalState.from_system(alchemy_system_vac)


    compound_state = CompoundThermodynamicState(therm_state, alchemical_state)
    compound_state_vac = CompoundThermodynamicState(therm_vac_state, alchemical_state_vac)

    compound_state.lambda_electrostatics = lambda_elec
    compound_state.lambda_sterics = lambda_sterics

    compound_state_vac.lambda_electrostatics = lambda_elec
    compound_state_vac.lambda_sterics = lambda_sterics

    integrator = LangevinMiddleIntegrator(T, 1.0/unit.picoseconds, stepSize)
    #causing issues to have one
    integrator_vac = LangevinMiddleIntegrator(T, 1.0/unit.picoseconds, stepSize)

    platform = Platform.getPlatformByName("CUDA")

    context = compound_state.create_context(integrator, platform)
    context.setPositions(system.get_positions())

    context_vac = compound_state_vac.create_context(integrator_vac, platform)

    simulation = app.Simulation(system.topology, alchemy_system, integrator, platform)
    simulation.context = context

    simulation_vac = app.Simulation(system_vac.topology, alchemy_system_vac, integrator_vac, platform)
    simulation_vac.context = context_vac

    reporter_solv = ThermodynamicDervivativesReporter(data_path, alchemy_system_vac, simulation_vac, 500)
    reporter_dcd = DCDReporter(dcd_path, 100)
    simulation.reporters.append(reporter_dcd)
    simulation.reporters.append(reporter_solv)



if __name__ == "__main__":
    base_file_path = "/work/users/r/d/rdey/BigBindDataset_New/"
    lig_index = str(521610)
    sim_path = "TD"
    curr_path = os.path.join(base_file_path, lig_index, sim_path)
    if(not os.path.exists(curr_path)):
        os.mkdir(curr_path)

    runSim(base_file_path, lig_index, sim_path, 1.0, 1.0, 100000, 1.0/unit.femtoseconds)

