from openmmtools.alchemy import AbsoluteAlchemicalFactory
from openmmtools import alchemy
from openmmtools.states import ThermodynamicState, CompoundThermodynamicState
from openmm import unit, LangevinMiddleIntegrator, Platform, app
from openmm.app.dcdreporter import DCDReporter
import openmm
from create_dataset import SolvDatasetReporter
from lr_complex import LRComplex, get_lr_complex
import os


class ThermodynamicDerivativesReporter(SolvDatasetReporter):
    """ 
    Subclass of SolvDatasetReporter that allows custom dp for finite difference in derivatives.
    """

    def __init__(self, filename, system, system_vac_context, report_interval):
        super().__init__(filename, system, report_interval)
        self.system_vac_context = system_vac_context

    def report(self, context, state):

        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)
        U = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        derivatives = state.getEnergyParameterDerivatives()

       
        print("Available derivative keys:", list(derivatives.values()))

        sterics_derivative = derivatives['lambda_sterics']
        electrostatics_derivative = derivatives['lambda_electrostatics']
        
        print("sterics_derivative:", sterics_derivative)
        print("electrostatics_derivative:", electrostatics_derivative)


        self.system_vac_context.setPositions(positions[self.mol_indices] * unit.nanometer)
        state_vac = self.system_vac_context.getState(getForces=True, getParameterDerivatives=True)

        vac_forces = state_vac.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)

        
        print("vac_forces:", vac_forces)

       
        vac_electrostatics_derivative = state_vac.getEnergyParameterDerivatives()['lambda_electrostatics']

        
        print("vac_electrostatics_derivative:", vac_electrostatics_derivative)

        lambda_sterics = context.getParameter('lambda_sterics')
        lambda_electrostatics = context.getParameter('lambda_electrostatics')

        
        print("lambda_sterics:", lambda_sterics)
        print("lambda_electrostatics:", lambda_electrostatics)





        self.file["positions"].resize((self.file["positions"].shape[0] + 1, len(self.mol_indices), 3))
        self.file["positions"][-1] = positions[self.mol_indices]

        self.file["solv_forces"].resize((self.file["solv_forces"].shape[0] + 1, len(self.mol_indices), 3))
        self.file["solv_forces"][-1] = forces[self.mol_indices] - vac_forces

        self.file["sterics_derivatives"].resize((self.file["sterics_derivatives"].shape[0] + 1,))
        self.file["sterics_derivatives"][-1] = sterics_derivative

        self.file["electrostatics_derivatives"].resize((self.file["electrostatics_derivatives"].shape[0] + 1,))
        #self.file["electrostatics_derivatives"][-1] = electrostatics_derivative.value_in_unit(unit.kilojoules_per_mole)
        self.file["electrostatics_derivatives"][-1] = (electrostatics_derivative - vac_electrostatics_derivative)

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
    reporter_step = 500
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
        consistent_exceptions=False,
        alchemical_pme_treatment='exact', 
        disable_alchemical_dispersion_correction=True,
        split_alchemical_forces=True
    )   

    system_region = alchemy.AlchemicalRegion(alchemical_atoms = system.lig_indices, annihilate_electrostatics = True, annihilate_sterics = False)
    system_vac_region = alchemy.AlchemicalRegion(alchemical_atoms = system_vac.lig_indices, annihilate_electrostatics = True, annihilate_sterics = False)

    alchemy_system = factory.create_alchemical_system(system.system, system_region)
    alchemy_system_vac = factory.create_alchemical_system(system_vac.system, system_vac_region)

    

    #I dont know why mmtools doesnt do this automatically??
    for force in alchemy_system.getForces():
        if (force.__class__ == openmm.CustomNonbondedForce or force.__class__ == openmm.CustomBondForce):
            for i in range(0, force.getNumGlobalParameters()):
                if (force.getGlobalParameterName(i) == "lambda_electrostatics"):
                    force.addEnergyParameterDerivative('lambda_electrostatics')
                elif (force.getGlobalParameterName(i) == "lambda_sterics"):
                    force.addEnergyParameterDerivative('lambda_sterics')

    for force in alchemy_system_vac.getForces():
        if (force.__class__ == openmm.CustomNonbondedForce or force.__class__ == openmm.CustomBondForce):
            for i in range(0, force.getNumGlobalParameters()):
                if (force.getGlobalParameterName(i) == "lambda_electrostatics"):
                    force.addEnergyParameterDerivative('lambda_electrostatics')
                elif (force.getGlobalParameterName(i) == "lambda_sterics"):
                    force.addEnergyParameterDerivative('lambda_sterics')




    therm_state = ThermodynamicState(alchemy_system, T)
    therm_vac_state = ThermodynamicState(alchemy_system_vac, T)

    therm_state.lambda_electrostatics = lambda_elec
    therm_state.lambda_sterics = lambda_sterics

    therm_vac_state.lambda_electrostatics = lambda_elec
    therm_vac_state.lambda_sterics = lambda_sterics
    
    alchemical_state = alchemy.AlchemicalState.from_system(alchemy_system)


    alchemical_state.lambda_sterics = lambda_sterics
    alchemical_state.lambda_electrostatics = lambda_elec

    alchemical_state_vac = alchemy.AlchemicalState.from_system(alchemy_system_vac)


    compound_state = CompoundThermodynamicState(thermodynamic_state=therm_state,
                                                   composable_states=[alchemical_state])
    compound_state_vac = CompoundThermodynamicState(thermodynamic_state = therm_vac_state, 
                                                        composable_states = [alchemical_state_vac])

    compound_state.lambda_electrostatics = lambda_elec
    compound_state.lambda_sterics = lambda_sterics

    compound_state_vac.lambda_electrostatics = lambda_elec
    compound_state_vac.lambda_sterics = lambda_sterics

    integrator = LangevinMiddleIntegrator(T, 2.0/unit.picoseconds, stepSize)

    integrator_vac = LangevinMiddleIntegrator(T, 2.0/unit.picoseconds, stepSize)

    platform = Platform.getPlatformByName("CUDA")

    context = compound_state.create_context(integrator, platform)
    context.setPositions(system.get_positions())

    context_vac = compound_state_vac.create_context(integrator_vac, platform)
    context_vac.setPositions(system_vac.get_positions())

    reporter_solv = ThermodynamicDerivativesReporter(data_path, system, context_vac, reporter_step)
    #reporter_dcd = DCDReporter(dcd_path, reporter_step)


    for _ in range(0, steps, reporter_step):
        integrator.step(reporter_step)
        state = context.getState(getPositions=True, 
                                getForces=True, 
                                getEnergy=True, 
                                getParameters=True, 
                                getParameterDerivatives = True)
        #reporter_dcd.report(context, state)
        reporter_solv.report(context, state)



if __name__ == "__main__":
    base_file_path = "/work/users/r/d/rdey/BigBindDataset_New/"
    lig_index = str(112014)
    sim_path = "TD"
    curr_path = os.path.join(base_file_path, lig_index, sim_path)
    if(not os.path.exists(curr_path)):
        os.mkdir(curr_path)

    runSim(base_file_path, lig_index, sim_path, 1.0, 1.0, 10000, 2.0*unit.femtoseconds)

