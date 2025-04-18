from copy import deepcopy
import json
import os
import subprocess
from traceback import print_exc
import pandas as pd
from config import CONFIG
from openmm import unit
from bigbind_solv.sim import SolvationSim
import time

# from configs import default
import sys

if __name__ == "__main__":
    lig_file = sys.argv[1]
    out_folder = sys.argv[2]
    solvent = sys.argv[3]
    equil_steps = int(sys.argv[4])
    sim = SolvationSim(lig_file, out_folder, solvent=solvent)
    sim.equil_steps = equil_steps
    sim.run_all()
    delta_F = sim.compute_delta_F()
    with open(out_folder + "/results.json", "w") as f:
        json.dump({
            "delta_F": float(delta_F),
            "elapsed_time": sim.elapsed_time,
        }, f)
