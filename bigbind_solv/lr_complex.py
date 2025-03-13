from copy import deepcopy
import os
import pickle

from tqdm import trange
import openmm as mm
import numpy as np
from rdkit import Chem
from openmm import app
from openmm import unit
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import EspalomaTemplateGenerator, GAFFTemplateGenerator
from openmmforcefields.generators import SystemGenerator
from openff.units.openmm import to_openmm


class LRComplex:
    """ Container for the OpenMM system for a receptor and/or ligand
    in complex, along with a bunch of helper functions.
    (We use this class for the receptor and ligand alone as well)"""

    def __init__(self,
                 system,
                 topology,
                 init_pos,
                 lig_indices,
                 pocket_indices=None,
                 cuda=True):
        """
        Parameters
        ----------
        system : openmm.System
            The OpenMM system for the receptor and ligand in complex
            
        topology : openmm.app.Topology
            The topology for the complex
            
        lig_indices : list of int
            The indices of the ligand atoms in the complex

        pocket_indices : list of int, optional, default=None
            The indices of the pocket atoms (+ the ligand atoms),
            sans alpha carbons. These will be assigned by default
            if None

        init_pos : np.ndarray
            The initial positions of the system

        cuda : bool, optional, default=True
            Whether to use CUDA or not
        """

        self.system = system
        self.topology = topology

        if os.environ.get('CUDA_VISIBLE_DEVICES', None) == "":
            cuda = False

        self.set_platform(cuda)

        self.lig_indices = np.array(lig_indices, dtype=int)
        self.rec_indices = np.ones(self.topology.getNumAtoms(), dtype=bool)
        self.rec_indices[self.lig_indices] = False

        self.make_simulation()
        self.set_positions(init_pos)

        if pocket_indices is None:
            positions = self.get_positions().value_in_unit(unit.angstrom)
            lig_pos = positions[self.lig_indices]
            self.pocket_indices = self.get_pocket_indices(lig_pos)
        else:
            self.pocket_indices = np.array(pocket_indices)

        # pocket indices + ligand indices
        self.all_pocket_indices = np.concatenate(
            [self.pocket_indices, self.lig_indices])

        self._diff_energy_system = None
        self._diff_energy_mask = None

        self.solvent_indices = np.zeros(self.system.getNumParticles(),
                                        dtype=bool)
        for res in self.topology.residues():
            if res.name in ("HOH", "WAT", "SOL", "CL", "Cl-", "NA", "Na+"):
                for atom in res.atoms():
                    self.solvent_indices[atom.index] = True

    def make_simulation(self):
        integrator = mm.LangevinIntegrator(300 * unit.kelvin,
                                           1.0 / unit.picoseconds,
                                           0.002 * unit.picoseconds)
        self.simulation = app.Simulation(self.topology, self.system,
                                         integrator, self.platform)

    # lmaoooo this legacy code from HS was killing performance -- this is actually pretty performant lol
    def set_platform(self, cuda):
        if cuda:
            self.platform = mm.Platform.getPlatformByName('CUDA')
            # self.platform.setPropertyDefaultValue('CudaPrecision', 'double')
        else:
            # self.platform = mm.Platform.getPlatformByName('Reference')
            self.platform = mm.Platform.getPlatformByName('CPU')

    def set_diff_energy_system(self, system, mask):
        """ I don't really like how we do this smh... but the
        idea is that you can set a system and a coordinate mask
        to subtract off the energy from part of this current system.
        Eg if you want to subtract off the energy from the intra-
        protein interactions. This only affects the get_potential_energy
        function. """
        self._diff_energy_system = system
        self._diff_energy_mask = mask

    def save(self, cache_prefix):
        """ Save to {cache_prefix}_system.xml and {cache_prefix}.pkl files
        (need two because OpenMM system's ain't picklable smh) """

        sys_file = f"{cache_prefix}_system.xml"
        other_file = f"{cache_prefix}.pkl"

        with open(sys_file, 'w') as f:
            f.write(mm.XmlSerializer.serialize(self.system))

        my_tup = self.get_tuple()
        with open(other_file, 'wb') as f:
            pickle.dump(my_tup, f)

    @classmethod
    def load(cls, cache_prefix, cuda=True):
        """ Load from {cache_prefix}_system.xml and {cache_prefix}.pkl files
        (need two because OpenMM system's ain't picklable smh) """

        sys_file = f"{cache_prefix}_system.xml"
        other_file = f"{cache_prefix}.pkl"

        with open(sys_file, 'r') as f:
            system = mm.XmlSerializer.deserialize(f.read())

        with open(other_file, 'rb') as f:
            args = pickle.load(f)

        return cls(system, *args, cuda=cuda)

    def get_tuple(self):
        """ Returns a tuple of everything needed to reconstruct the system
        (except for the system itself, which is saved separately) """
        return (self.topology, self.get_positions(), self.lig_indices,
                self.pocket_indices)

    def copy(self):
        system, tup = deepcopy((self.system, self.get_tuple()[1:]))
        # not deepcopying the topology because weird errors when trying
        # to save to PDB files (it seems like deepcopying destroys the
        # internal references to the atoms smh)
        return LRComplex(system, self.topology, *tup)

    def set_positions(self, positions, box_vectors=None):
        """ Set the positions of the system. If box_vectors is provided, set them as well """
        # smh so OpenMM assumes the units are nm -- this is bad.
        # We raise errors if you don't provide units
        if not unit.is_quantity(positions):
            raise ValueError("Positions must have a unit attached")
        self.simulation.context.setPositions(positions)
        if box_vectors is not None:
            a = mm.Vec3(*box_vectors[0])
            b = mm.Vec3(*box_vectors[1])
            c = mm.Vec3(*box_vectors[2])
            self.simulation.context.setPeriodicBoxVectors(a, b, c)

    def get_positions(self):
        """ Get the positions of the system """
        return self.simulation.context.getState(
            getPositions=True).getPositions(asNumpy=True)

    def get_potential_energy(self):
        """ Get the potential energy of the system """
        U = self.simulation.context.getState(
            getEnergy=True).getPotentialEnergy().in_units_of(
                unit.kilocalorie_per_mole)
        if self._diff_energy_system is not None:
            self._diff_energy_system.set_positions(
                self.get_positions()[self._diff_energy_mask])
            diff_U = self._diff_energy_system.get_potential_energy()
            U -= diff_U
        return U

    def get_lig_positions(self):
        """ Get the positions of the ligand """
        return self.get_positions()[self.lig_indices]

    def get_rec_positions(self):
        """ Get the positions of the receptor """
        return self.get_positions()[self.rec_indices]

    def get_forces(self):
        """ Get the forces on the system """
        return self.simulation.context.getState(getForces=True).getForces(
            asNumpy=True)

    def save_positions(self, filename):
        """ Save the positions of the system to a pkl file """
        positions = self.get_positions().value_in_unit(unit.angstrom)
        with open(filename, 'wb') as f:
            pickle.dump(positions, f)

    def load_positions(self, filename):
        """ Load positions from a pkl file """
        with open(filename, 'rb') as f:
            positions = pickle.load(f) * unit.angstrom
        self.set_positions(positions)

    def minimize_energy(self, tol=5e-3, quiet=False):
        """ Minimize the energy up to tol (in kcal/mol) """

        tol = tol * unit.kilocalorie_per_mole
        if not quiet:
            print('Minimizing energy...')
        self.simulation.minimizeEnergy(
            tolerance=tol.value_in_unit(unit.kilojoule_per_mole))

        state = self.simulation.context.getState(getEnergy=True)
        U = state.getPotentialEnergy()
        if not quiet:
            print(
                f"Minimized energy: {U.value_in_unit(unit.kilocalorie_per_mole)} kcal/mol"
            )

    def minimize_cached(self, cache_filename, tol=5e-3):
        """ Minimization can take a while; this function allows
        us to cache the results  """
        try:
            self.load_positions(cache_filename)
        except (FileNotFoundError, EOFError):
            self.minimize_energy(tol)
            self.save_positions(cache_filename)

    def minimize_force_tol(self, force_tol=0.019, max_iter=150):
        """ Minimize the energy until the mean force on the atoms
        is less than force_tol (in kcal/mol/angstrom) """

        tol = 0 * unit.kilocalorie_per_mole
        print('Minimizing energy...')
        for i in range(max_iter):
            self.simulation.minimizeEnergy(tolerance=tol.value_in_unit(
                unit.kilojoule_per_mole),
                                           maxIterations=10000)
            forces = self.get_forces().value_in_unit(
                unit.kilocalorie_per_mole / unit.angstrom)
            mean_force = np.mean(np.linalg.norm(forces, axis=1))
            if mean_force < force_tol:
                break

        state = self.simulation.context.getState(getEnergy=True)
        U = state.getPotentialEnergy()
        print(
            f"Minimized energy: {U.value_in_unit(unit.kilocalorie_per_mole)} kcal/mol"
        )

    def minimize_force_tol_cached(self,
                                  cache_filename,
                                  force_tol=0.019,
                                  max_iter=150):
        """ Minimization can take a while; this function allows
        us to cache the results  """
        if os.path.exists(cache_filename):
            self.load_positions(cache_filename)
        else:
            self.minimize_force_tol(force_tol, max_iter)
            self.save_positions(cache_filename)

    def minimize_anneal(self, Tmax, n_iter):
        """ Uses basic simulated annealing to minimize the energy,
        hopefully getting over some small energy barriers.
        Note: assumes energy has already been minimized.
        
        Parameters
        ----------
        Tmax : unit.Quantity
            The maximum temperature to use in the annealing
            
        n_iter : int
            The number of iterations to use in the annealing. The
            simulation steps for 100 steps at each temperature """

        # initial minimization with a high tolerance
        self.minimize_energy(tol=5)

        T = Tmax
        t = trange(n_iter)
        for i in t:
            T = (n_iter - i) * Tmax / n_iter
            self.simulation.integrator.setTemperature(T)
            self.simulation.step(100)
            state = self.simulation.context.getState(getEnergy=True)
            U = state.getPotentialEnergy()
            t.set_description(
                f"T: {T.value_in_unit(unit.kelvin):0.2f}K, U: {U.value_in_unit(unit.kilocalorie_per_mole):0.2f} kcal/mol"
            )

        self.minimize_force_tol()

    def minimize_anneal_cached(self, cache_filename, Tmax, n_iter):
        """ Minimization can take a while; this function allows
        us to cache the results  """

        # cache_filename = f"{cache_prefix}_Tmax{Tmax.value_in_unit(unit.kelvin):0.2f}_niter{n_iter}.pkl"

        if os.path.exists(cache_filename):
            self.load_positions(cache_filename)
        else:
            self.minimize_anneal(Tmax, n_iter)
            self.save_positions(cache_filename)

    def save_to_pdb(self, filename):
        """ Save the system to a PDB file """
        positions = self.get_positions()
        app.PDBFile.writeFile(self.topology, positions, open(filename, 'w'))

    def get_pocket_indices(self,
                           lig_pos,
                           cutoff_dist=3.0,
                           excluded_atoms=["N", "C", "O"]):
        """ Returns indices of all the atoms in the
        pocket residues (defined by the cutoff distance to the ligand)
        By default it only includes sidechain atoms (not the backbone) """

        if len(lig_pos) == 0:
            # if no ligand, no defined pocket
            return np.array([])

        positions = self.get_positions().value_in_unit(unit.angstrom)

        residues = list(self.topology.residues())
        pocket_residues = []
        for r in residues:
            for atom in r.atoms():
                dists = np.linalg.norm(positions[atom.index] - lig_pos, axis=1)
                min_dist = np.min(dists)
                if min_dist < 0.01:
                    continue  # ligand atom
                if min_dist < cutoff_dist:
                    pocket_residues.append(r)
                    break

        poc_sidechain_indices = []
        for residue in pocket_residues:
            for atom in residue.atoms():
                if atom.name in excluded_atoms:
                    continue
                # print(residue.name, atom.name, rec.GetAtomWithIdx(atom.index).GetSymbol())
                poc_sidechain_indices.append(atom.index)

        return np.array(poc_sidechain_indices)

    def has_ligand(self):
        """ Returns True if the system has a ligand """
        return len(self.lig_indices) > 0

    def has_rec(self):
        """ Returns True if the system has a receptor """
        return len(self.lig_indices) < self.topology.getNumAtoms()

    def get_adjacency_matrix(self):
        """ Returns the adjacency matrix for the system """
        n_atoms = self.topology.getNumAtoms()
        adj = np.zeros((n_atoms, n_atoms), dtype=bool)
        for bond in self.topology.bonds():
            i = bond.atom1.index
            j = bond.atom2.index
            adj[i, j] = 1
            adj[j, i] = 1
        return adj


def get_lr_complex(prot_pdb,
                   lig_sdf,
                   lig_index=0,
                   pocket_indices=None,
                   lig_ff="gaff",
                   solvent="obc2",
                   constraints=app.HBonds,
                   nonbonded_method=app.NoCutoff,
                   nonbonded_cutoff=0.9 * unit.nanometer,
                   include_barostat=False,
                   box_vectors=None,
                   box_padding=1.6 * unit.nanometer,
                   P=1.0 * unit.atmosphere,
                   T=300 * unit.kelvin,
                   cache_dir=None,
                   extra_mols=[]):
    """ Loads an LRComplex from a protein PDB and a ligand SDF;
    lig_index indiciates which ligand in the SDF to use. Both
    prot_pdb and lig_sdf are optional. Note that it is assumed
    that pdbfixer has been run on the protein PDB and the ligand
    has hydrogens added (in the correct protonation state).
    
    lig_ff is how we parameterize the ligand. Currently supports GAFF and
    Espaloma (espaloma-0.3.2).

    Solvent is either "tip[n]p", "gbn[1-2]", "obc[1-2]", or "none". If "none", then
    we are in a vacuum. If "tip3p", then we are in explicit solvent and this
    will create the solvent box.

    """

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    prot = None
    lig = None

    modeller = app.Modeller(app.Topology(), [])
    if prot_pdb is not None:
        prot = app.PDBFile(prot_pdb)
        modeller.add(prot.topology, prot.positions)
    if lig_sdf is not None:

        lig = Chem.SDMolSupplier(lig_sdf)[lig_index]
        if lig is None:
            raise ValueError(f"Could not load ligand from {lig_sdf}")
        lig = Molecule.from_rdkit(lig)
        top = lig.to_topology().to_openmm()
        for residue in top.residues():
            residue.name = "LIG"

        modeller.add(top, to_openmm(lig.conformers[0]))

    forcefield_kwargs = {
        "nonbondedMethod": nonbonded_method,
        "nonbondedCutoff": nonbonded_cutoff,
        "constraints": constraints,
    }

    ffs = ['amber/ff14SB.xml']
    if solvent != "none":
        prefix = "amber14" if solvent == "tip3p" else "implicit"
        ffs.append(f'{prefix}/{solvent}.xml')

    mols = [lig] if lig is not None else []
    for mol_file in extra_mols:
        mols.append(Molecule.from_file(mol_file))

    forcefield = app.ForceField(*ffs)
    if lig_ff == "gaff":
        generator = GAFFTemplateGenerator(
            mols,
            cache=None if cache_dir is None else f"{cache_dir}/gaff.json")
    elif lig_ff == "espaloma":
        generator = EspalomaTemplateGenerator(
            mols,
            cache=None if cache_dir is None else f"{cache_dir}/espaloma.json",
            forcefield='espaloma-0.3.2')
    else:
        raise ValueError(f"lig_ff must be gaff or espaloma, not {lig_ff}")
    forcefield.registerTemplateGenerator(generator.generator)

    residues = list(modeller.topology.residues())
    lig_indices = []
    if lig is not None:
        for atom in residues[-1].atoms():
            lig_indices.append(atom.index)
        lig_indices = np.array(lig_indices)

    if solvent == "tip3p":
        modeller.addSolvent(
            forcefield,
            model='tip3p',
            padding=box_padding,
            positiveIon='Na+',
            negativeIon='Cl-',
            ionicStrength=0.0 * unit.molar,
            neutralize=True,
        )

    if box_vectors is not None:
        modeller.topology.setPeriodicBoxVectors(box_vectors)

    system = forcefield.createSystem(modeller.topology, **forcefield_kwargs)

    if include_barostat:
        system.addForce(mm.MonteCarloBarostat(P, T))

    return LRComplex(system, modeller.topology, modeller.positions,
                     lig_indices, pocket_indices)


def get_all_pocket_indices(rec_file,
                           lig_files,
                           cutoff_dist=3.0,
                           excluded_atoms=["N", "C", "O"]):
    """ Returns the indices of the pocket atoms that
    are within cutoff_dist angstroms of _any_ conformer of _any_
    ligand in lig_files. """
    rec_sys = get_lr_complex(rec_file, None)

    lig_pos = []
    for lig_file in lig_files:
        ligs = Molecule.from_file(lig_file)
        if not isinstance(ligs, list):
            ligs = [ligs]
        for lig in ligs:
            for conformer in lig.conformers:
                lig_pos += conformer._magnitude.tolist()
    lig_pos = np.array(lig_pos)
    return rec_sys.get_pocket_indices(lig_pos,
                                      cutoff_dist=cutoff_dist,
                                      excluded_atoms=excluded_atoms)
