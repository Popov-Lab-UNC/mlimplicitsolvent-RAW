import subprocess

from tqdm import tqdm
from datasets.bigbind_solv import BigBindSolvDataset
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from openff.toolkit.topology import Molecule
from openmm import app
from openmm import unit
import openmm as mm
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.units.openmm import to_openmm
from datasets.md_batch import *


def to_xyz_block(data):
    """Converts a MDData object to a string in xyz format"""
    table = Chem.GetPeriodicTable()
    positions = data.positions * 10
    elems = [table.GetElementSymbol(int(num)) for num in data.atomic_numbers]

    out = ""
    out += f"{len(elems)}\n\n"
    for i in range(len(elems)):
        out += f"{elems[i]} {positions[i][0]} {positions[i][1]} {positions[i][2]}\n"

    return out


def to_rdkit(data):
    """Converts a MDData object to a RDKit molecule"""
    xyz_block = to_xyz_block(data)
    mol = Chem.MolFromXYZBlock(xyz_block)
    total_charge = round(float(data.charges.sum()))
    rdDetermineBonds.DetermineBonds(mol, charge=total_charge)
    return mol


def to_off_mol(data):
    """Converts a MDData object to a OpenFF molecule. Makes sure to set
    the charges as well."""
    mol = to_rdkit(data)
    off_mol = Molecule.from_rdkit(mol)
    charges = data.charges.numpy() * unit.elementary_charge
    off_mol.partial_charges = charges

    return off_mol

def to_openmm_topology(data):
    """Converts a MDData object to a OpenMM topology"""
    off_mol = to_off_mol(data)
    return off_mol.to_topology().to_openmm()

def to_openmm_system(
    data,
    solvent="obc2",
    box_padding=1.6 * unit.nanometer,
    constraints=app.HBonds,
    P=1.0 * unit.atmosphere,
    T=300.0 * unit.kelvin,
    nonbonded_cutoff=0.9*unit.nanometer,
):
    """Creates an OBC2 system from a MDData object"""
    off_mol = to_off_mol(data)

    topology = off_mol.to_topology().to_openmm()
    positions = to_openmm(off_mol.conformers[0])
    modeller = app.Modeller(topology, positions)

    ffs = ["amber/ff14SB.xml"]
    if solvent != "none":
        prefix = "amber14" if solvent == "tip3p" else "implicit"
        ffs.append(f"{prefix}/{solvent}.xml")

    generator = GAFFTemplateGenerator(molecules=[off_mol])
    forcefield = app.ForceField(*ffs)
    forcefield.registerTemplateGenerator(generator.generator)

    if solvent == "tip3p":
        modeller.addSolvent(
            forcefield,
            model="tip3p",
            padding=box_padding,
            positiveIon="Na+",
            negativeIon="Cl-",
            ionicStrength=0.0 * unit.molar,
            neutralize=True,
        )
        nonbonded_method = app.PME
    else:
        nonbonded_method = app.NoCutoff

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=nonbonded_method,
        constraints=constraints,
        nonbondedCutoff=nonbonded_cutoff,
    )

    if solvent == "tip3p" and P is not None and T is not None:
        system.addForce(mm.MonteCarloBarostat(P, T))

    return system, modeller

def get_gb_forces(data):
    system, modeller = to_openmm_system(data)
    topology = modeller.topology

    for force in system.getForces():
        if isinstance(force, mm.CustomGBForce):
            force.setForceGroup(1)

    integrator = mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    simulation = app.Simulation(topology, system, integrator)

    pos = data.positions.numpy()*unit.nanometers
    simulation.context.setPositions(pos)

    gb_forces = simulation.context.getState(getForces=True, groups=1<<1).getForces(asNumpy=True)
    gb_forces = gb_forces.value_in_unit(unit.kilojoules_per_mole/unit.nanometer)

    return gb_forces

def main():
    dataset = BigBindSolvDataset("val", 0)
    Y_true = []
    Y_pred = []
    for data in tqdm(dataset):
        if len(Y_true) > 20: # change this! Use all the values
            break
        # we want to get rid of this -- add lambda parameters to GB
        if data.lambda_electrostatics == 1 and data.lambda_sterics == 1:
            try:
                gb_forces = get_gb_forces(data)
            except ValueError:
                continue
            Y_true.append(data.forces.numpy())
            Y_pred.append(gb_forces)
            # need to save the forces
    
    # now evaluate metrics on Y_true and Y_pred