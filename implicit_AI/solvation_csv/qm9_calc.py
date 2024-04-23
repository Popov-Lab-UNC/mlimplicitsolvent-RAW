print("qm9 Solvation Energy 1")
from rdkit import Chem
print("qm9 Solvation Energy 2")
from rdkit.Chem import *
print("qm9 Solvation Energy 3")
from openmm.unit import *
print("qm9 Solvation Energy 4")
from openmm import *
print("qm9 Solvation Energy 5")
from openmm.app import *
print("qm9 Solvation Energy 6")
from openff.toolkit.topology import Molecule
print("qm9 Solvation Energy 7")
from openmmforcefields.generators import GAFFTemplateGenerator
print("qm9 Solvation Energy 8")
from openff.units.openmm import to_openmm 
print("qm9 Solvation Energy 9")
import pandas as pd 

print("qm9 Solvation Energy 10")

def o_simulation_run(system, mol_t, mol_p):

    integrator = VerletIntegrator(1.0 * unit.femtosecond)
    simulation = Simulation(mol_t, system, integrator)
    simulation.context.setPositions(mol_p)
    simulation.minimizeEnergy()
    state = simulation.context.getState(getForces=True, groups=1)
    force = state.getForces(asNumpy = True)
    
    return force

def openmm_solv(i, File):
    if ligand is None:
        print(f"Error: Unable to load ligand from SDF: Ligand{i}.")
        return
    name = f"gdb9_{i+1}"
    success = True
    try:
        print(name)
        molecule = Molecule.from_rdkit(File)
        mol_t = molecule.to_topology().to_openmm()
        mol_p = to_openmm(molecule.conformers[0]) 
        gaff = GAFFTemplateGenerator(molecules=molecule)
        forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
        forcefield.registerTemplateGenerator(gaff.generator)
        system = forcefield.createSystem(mol_t)
        for i in range(system.getNumForces()):
            force = system.getForce(i)
            if isinstance(force, CustomGBForce):
                Force.setForceGroup(force, 1)
                break
        
        solv_force = o_simulation_run(system, mol_t, mol_p)
        input = [solv_force]
        return 
    except Exception as e:
        input = [name, f"Failed on openmm_solv{i}: {e}"]
        success= False
    finally:
        print(input)
        return input, success

def ligands_load(file):
    # Load the ligand structure from SDF using RDKit
    ligand_supplier = Chem.SDMolSupplier(file, removeHs= False, sanitize= False)
    # Check if the ligand was successfully loaded
    if ligand_supplier is None or len(ligand_supplier) == 0:
        print("Error: Unable to load ligand from SDF.")
        exit()
    return ligand_supplier

if __name__ == "__main__":
    print("Solvation Calculation - QM9 Dataset")
    m, n, i = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    file = "../raw/gdb9.sdf"
    output = []
    fail = 0
    ligand_supplier = ligands_load(file)
    for idx, ligand in enumerate(ligand_supplier):
        if (m <= idx < n):
            input, success = openmm_solv(idx, ligand)
            output.append(input)
            if(success != True):
                fail+=1
    failure_rate = round(fail/(n-m), 5)
    print(f"Failure rate of {failure_rate}")
    db = pd.DataFrame(output)
    db.to_csv(f"solvation{i}.csv")

