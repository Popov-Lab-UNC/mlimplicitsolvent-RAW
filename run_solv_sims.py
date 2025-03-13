from copy import deepcopy
import os
import subprocess
import pandas as pd
from config import CONFIG
from openmm import unit
from bigbind_solv.sim import SolvationSim
import time

# from configs import default
import sys
from bigbind_solv.freesolv import load_freesolv, smi_to_protonated_sdf


def run_all_sims(solvent, equil_steps, freesolv_split):

    os.makedirs("output", exist_ok=True)

    df = pd.read_csv(f"freesolv/SAMPL{freesolv_split}.csv")
    with open(f"output/freesolv_results_{solvent}_{equil_steps}.csv", "w") as f:
        f.write("iupac,delta_F,exp_dG,calc_dG,elapsed_time\n")

        for i, row in df.iterrows():
            out_folder = os.path.join(
                CONFIG.cache_dir, f"freesolv_{solvent}_{equil_steps}", str(i)
            )
            os.makedirs(out_folder, exist_ok=True)

            print(f"Processing {row.iupac}")
            print(f"Saving results to {out_folder}")

            lig_file = os.path.join(out_folder, "ligand.sdf")
            if not os.path.exists(lig_file):
                smi_to_protonated_sdf(row.smiles, lig_file)

            sim = SolvationSim(lig_file, out_folder, solvent=solvent)
            sim.equil_steps = equil_steps
            sim.run_all()
            delta_F = sim.compute_delta_F()
            elapsed_time = sim.elapsed_time

            print(
                f"Computed delta_F: {delta_F.value_in_unit(unit.kilocalories_per_mole)} kcal/mol, took {elapsed_time} seconds"
            )
            print(row)

            f.write(
                f"{row.iupac},{delta_F.value_in_unit(unit.kilocalories_per_mole)},{row.exp_dG},{row.calc_dG},{elapsed_time}\n"
            )
            f.flush()

def split_freesolv():
    df_og = pd.read_csv("freesolv/SAMPL.csv")
    df = deepcopy(df_og)

    # split into 4 sets
    df1 = df.sample(frac=0.25)
    df = df.drop(df1.index)
    df2 = df.sample(frac=0.33)
    df = df.drop(df2.index)
    df3 = df.sample(frac=0.5)
    df4 = df.drop(df3.index)
    df1.to_csv("freesolv/SAMPL0.csv", index=True)
    df2.to_csv("freesolv/SAMPL1.csv", index=True)
    df3.to_csv("freesolv/SAMPL2.csv", index=True)
    df4.to_csv("freesolv/SAMPL3.csv", index=True)

    assert len(df_og) == len(df1) + len(df2) + len(df3) + len(df4)

if __name__ == "__main__":
    solvent = str(sys.argv[1])
    equil_steps = int(sys.argv[2])
    freesolv_split = int(sys.argv[3])
    run_all_sims(solvent, equil_steps, freesolv_split)
