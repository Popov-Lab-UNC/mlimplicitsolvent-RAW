from epsilon_calculation import runSim, SolvDatasetReporterWithCustomDP
from openmm import app, unit, LangevinMiddleIntegrator, NonbondedForce
from openmm.app.dcdreporter import DCDReporter
from freesolv import smi_to_protonated_sdf
from sim import make_alchemical_system
from fep import set_fep_lambdas
from config_dict import config_dict
from lr_complex import LRComplex, get_lr_complex
import os
import mdtraj as md
import random
import shutil
import numpy as np
from tqdm import tqdm
import h5py
from openmm.app.internal.customgbforces import GBSAGBn2Force
import traceback
import sys


def runAll(folder, smile, name, start, end):
    sdf_file = os.path.join(folder, f"{name}.sdf")
    init_steps = 3000000
    lambda_steps = 250000
    dp = 1e-4
    if (not os.path.exists(sdf_file)):
        smi_to_protonated_sdf(smile, sdf_file)
    kwargs = {
        "nonbonded_cutoff": 0.9 * unit.nanometer,
        # "nonbonded_cutoff": 1.5*unit.nanometer,
        "constraints": app.HBonds,
        "box_padding": 1.6 * unit.nanometer,
        # "box_padding": 2.0*unit.nanometer,
        "lig_ff": "gaff",
        "cache_dir": folder,
    }
    system = get_lr_complex(None,
                            sdf_file,
                            solvent="tip3p",
                            nonbonded_method=app.PME,
                            include_barostat=True,
                            **kwargs)
    system_vac = get_lr_complex(None, sdf_file, solvent="none", **kwargs)
    system.save(os.path.join(folder, "system"))
    system_vac.save(os.path.join(folder, "system_vac"))

    system.save_to_pdb(os.path.join(folder, "system.pdb"))
    system_vac.save_to_pdb(os.path.join(folder, "system_vac.pdb"))

    dcd_file = os.path.join(folder, "init.dcd")

    if (not os.path.exists(os.path.join(folder, 'init.dcd'))):
        integrator = LangevinMiddleIntegrator(300 * unit.kelvin,
                                              1.0 / unit.picosecond,
                                              2.0 * unit.femtosecond)

        #No need for initial system to be alchemical
        simulation = app.Simulation(system.topology, system.system, integrator,
                                    system.platform)
        simulation.reporters = []
        dcd_reporter = DCDReporter(file=dcd_file, reportInterval=100)
        simulation.context.setPositions(system.get_positions())
        simulation.minimizeEnergy()
        simulation.reporters.append(dcd_reporter)

        simulation.step(init_steps)

    #Use conformations generated to calculate other MAFs
    traj = md.load(dcd_file, top=os.path.join(folder, "system.pdb"))
    vac_indicies = system.lig_indices
    for i in vac_indicies:
        system.system.setParticleMass(i, 0.0)

    alchemical_system_vac = make_alchemical_system(system_vac)

    for idx, coords in tqdm(enumerate(traj.xyz),
                            total=len(traj.xyz),
                            desc="Processing trajectory"):
        if (not (start <= idx < end)):
            continue
        try:
            frame_folder = os.path.join(folder, str(idx))
            sim_path = os.path.join(frame_folder, 'sim.h5')

            if (os.path.exists(frame_folder)):
                shutil.rmtree(frame_folder)

            os.mkdir(frame_folder)

            alchemical_system = make_alchemical_system(system)
            alchemical_system.set_positions(coords * unit.nanometer)

            lambda_electrostatics, lambda_sterics = random_lambda()

            set_fep_lambdas(alchemical_system.simulation.context,
                            lambda_sterics, lambda_electrostatics)
            set_fep_lambdas(alchemical_system_vac.simulation.context,
                            lambda_sterics, lambda_electrostatics)

            vac_coords = np.array(coords)[vac_indicies]
            alchemical_system_vac.set_positions(vac_coords.tolist() *
                                                unit.nanometer)

            solv_reporter = SolvDatasetReporterWithCustomDP(
                sim_path, alchemical_system_vac, 500, dp)

            simulation = alchemical_system.simulation
            simulation.reporters.append(solv_reporter)
            simulation.step(lambda_steps)
        except Exception as e:
            print(f"Exception has occured: {str(e)}")


def collate(folder, name):
    idx = np.arange(0, 30000)
    np.random.shuffle(idx)
    train = idx[:24000]
    val = idx[24000:27000]
    test = idx[27000:]
    min_start_frame = 10
    name_dict = {'train': train, 'val': val, 'test': test}
    for k, v in name_dict.items():
        out_folder = os.path.join(folder, f'{k}.h5')
        out_file = h5py.File(out_folder, "w")
        for idx in v:
            try:
                f_folder = os.path.join(folder, str(idx))
                f_file = os.path.join(f_folder, 'sim.h5')
                file = h5py.File(f_file, 'r')

                for key in file.keys():
                    if np.isnan(file[key]).sum() > 0:
                        raise RuntimeError("Found NaNs!")

                if np.any(np.abs(file["solv_forces"]) > 10000):
                    raise RuntimeError("Forces are too large")

                group = out_file.create_group(str(idx))
                for key in file.keys():
                    group.create_dataset(key, data=file[key][min_start_frame:])
                    print(group[key])

                sys_prefix = folder + "/system_vac"
                complex = LRComplex.load(sys_prefix)
                atomic_numbers = np.array([
                    atom.element.atomic_number
                    for atom in complex.topology.atoms()
                ])

                for force in complex.system.getForces():
                    if isinstance(force, NonbondedForce):
                        break
                else:
                    raise RuntimeError("Failed to find Nonbonded force")

                charges = []
                for i in range(complex.system.getNumParticles()):
                    q, *_ = force.getParticleParameters(i)
                    charges.append(q.value_in_unit(unit.elementary_charge))
                charges = np.array(charges, dtype=np.float32)

                assert len(charges) == len(atomic_numbers)

                group["atomic_numbers"] = atomic_numbers
                group["charges"] = charges

                # now get the GB forces and energies...

                lig_file = os.path.join(folder, f"{name}.sdf")

                kwargs = {
                    "nonbonded_cutoff": 0.9 * unit.nanometer,
                    # "nonbonded_cutoff": 1.5*unit.nanometer,
                    "constraints": app.HBonds,
                    "box_padding": 1.6 * unit.nanometer,
                    # "box_padding": 2.0*unit.nanometer,
                    "lig_ff": "gaff",
                }
                complex_obc = get_lr_complex(None,
                                             lig_file,
                                             solvent="obc2",
                                             **kwargs)
                force = GBSAGBn2Force(cutoff=None,
                                      SA="ACE",
                                      soluteDielectric=1,
                                      solventDielectric=78.5)
                gnn_params = np.array(
                    force.getStandardParameters(complex_obc.topology))
                group["gnn_params"] = gnn_params
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(str(e))

                traceback.print_exc()


def random_lambda(full_frac=0.2):

    if random.random() < full_frac:
        lambda_sterics = 0.99999973
        lambda_electrostatics = 0.9999973
    else:
        # alwaays remove electrostatics before sterics
        if random.random() < 0.5:
            lambda_sterics = random.uniform(2.7e-6, 0.9999973)
            lambda_electrostatics = 2.7e-6
        else:
            lambda_sterics = 0.9999973
            lambda_electrostatics = random.uniform(2.7e-6, 0.9999973)
    return lambda_electrostatics, lambda_sterics


if __name__ == '__main__':
    smile = "CC(C)C=C"
    folder = config_dict['single_molecule']
    name = '3-methylbut-1-ene'
    #start = int(sys.argv[1])
    #end = int(sys.argv[2])
    #runAll(folder, smile, name, start, end)
    collate(folder, name)
