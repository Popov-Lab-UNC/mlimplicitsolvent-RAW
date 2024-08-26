import os
from openff.toolkit.topology import Molecule, Topology
from openmmforcefields.generators import GAFFTemplateGenerator
from openmmtorch import TorchForce
from openmm import GBSAGBn2Force
import torch
from openmm import app
from openmm.app import *
from alchemlyb.estimators import MBAR
import numpy as np
from openmm.unit import *
from openmm import LangevinMiddleIntegrator
from openmm import StateDataReporter
from openmmtools.constants import kB

class DataConnection:
    def __init__(self, positions, lambda_electrostatics, lambda_sterics):
        self.pos = positions
        self.lambda_electrostatics = lambda_electrostatics
        self.lambda_sterics = lambda_sterics

class AI_Solvation_calc:

    @staticmethod
    def get_ligand_and_water_indices(system):
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
    
    @staticmethod
    def create_system_from_smiles(smiles, lambda_electrostatics):
        molecule = Molecule.from_smiles(smiles)
        for atom, p_charge in zip(molecule.atoms, molecule.partial_charges):
            atom.partial_charge = p_charge*lambda_electrostatics
        gaff = GAFFTemplateGenerator(molecule=molecule)
        forcefield = app.Forcefield()
        forcefield.registerTemplateGenerator(gaff.generator)
        system = forcefield.createSystem(molecule.topology, nonBondedMethod = app.NoCutoff)
        return system, Molecule

    

    def __init__(self, model, smiles):
        self.lambda_sterics = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.lambda_electrostatics = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.model = torch.load(model)
        self.model.eval()
        self.n_steps = 10000
        self.energies = []
        self.energies_vac = []
        self.smiles = smiles
        self.report_interval = 1000


    def set_model(self):
        self.set_gnn_params()
        self.model_force = TorchForce(self.model)
        self.model_force.addGlobalParameter("lambda_sterics", 1.0)
        self.model_force.addGlobalParameter("lambda_electrostatics", 1.0)
        self.model_force.addGlobalParameter("vaccum", 0.0)
        self.model_force.addGlobalParameter('testing', 0.0)
        self.model_force.setOutputsForces(True)

    def set_gnn_params(self):
        assert self.system
        self.model.set_gnn_params(self.compute_atom_features)

    def set_system(self, lambda_electrostatics, cached = None, ):
        if cached:
            self.system = cached #Finish it later
        else:
            self.system, self.molecule = self.create_system_from_smiles(self.smiles, lambda_electrostatics=lambda_electrostatics)
            self.topology = self.molecule.to_topology()

    def compute_atom_features(self):
        charges = np.array([tuple(f.getParticleParameters)[0] for f in self.system.getForces() if isinstance(f, App.NonbondedForce)][0])
        charges = np.array([self.system.getForces()[0].getParticleParameters(i)[0]._value for i in range(self.topology._numAtoms)])

        force = GBSAGBn2Force(cutoff=None,SA="ACE",soluteDielectric=1,solventDielectric=4)
        gbn2_parameters = np.empty((self.topology.getNumAtoms(),6))
        gbn2_parameters[:,0] = charges # Charges
        gbn2_parameters[:,1:] = force.getStandardParameters(self.topology)
        return gbn2_parameters
    
    def AI_simulation(self, lambda_sterics, lambda_electrostatics, vaccum):
        self.set_system(lambda_electrostatics)
        self.system.addForce(self.model_force)
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/unit.picosecond, 0.004*unit.picoseconds)
        simulation = Simulation(self.topology, self.system, integrator)
        simulation.context.setParameter("vaccum", vaccum)
        simulation.context.setParameter("lambda_electrostatics", lambda_electrostatics)
        simulation.context.setParameter("lambda_sterics", lambda_sterics)
        simulation.minimizeEnergy()
        if vaccum: 
            reporter = StateDataReporter(f"{self.smiles}_energy.txt", self.report_interval, potentialEnergy = True, temperature=True)
            simulation.reporters.append(reporter)

        simulation.step(self.n_steps)

    
    def run_all_sims(self):
        self.set_system(1.0)
        self.set_model()
        self.AI_simulation(1.0, 1.0)

        print("Removing electrostatics")
        for lambda_elec in reversed(self.lambda_electrostatics):
            self.AI_simulation(1.0, lambda_elec, vaccum = False)
        print("Removing sterics")
        for lambda_ster in reversed(self.lambda_sterics):
            self.AI_simulation(lambda_ster, 0.0, vaccum = False)
        print("Re-adding electrostatics")
        if not os.path.exists(f"{self.smiles}_energy.txt"):
            for lambda_elec in self.lambda_electrostatics:
                self.AI_simulation(0.0, lambda_elec, vaccum = True)

    def compute_delta_F(self):
        solv = self.model.energies 
        with open(f"{self.smiles}_energy.txt", 'r') as file:
             combine = np.array([float(line.strip().split(" ")) for line in file])
             vac = combine[:,0]/(kB*T)
             T = combine[:,1]*unit.kelvin
        mbar_vac = MBAR()
        mbar_vac.fit(vac)

        mbar_solv = MBAR()
        mbar_solv.fit(solv) 

        F_solv_kt = mbar_vac.delta_f_[0][1] - mbar_solv.delta_f_[(0,0)][(1,1)]
        F_solv = F_solv_kt*T*kB

        return F_solv

        
        




    




