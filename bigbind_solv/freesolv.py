### We have saved the freesolv database locally

import os
import subprocess
import pandas as pd
from openmm import unit

from config import CONFIG

def load_freesolv():
    return pd.read_csv("data/freesolv.txt", delimiter=";", comment="#", names=["compound", "smiles", "compound_name", "exp_dG", "exp_uncertainty", "calc_dG", "calc_uncertainty", "exp_ref", "cacl_ref", "notes"])

def smi_to_protonated_sdf(smi, out_file):
    """ Uses openbabl to add a conformer + protonate the molecule, saving to out_file """
    cmd = f"obabel -:'{smi.strip()}' -O {out_file} --gen3d --pH 7"
    subprocess.run(cmd, check=True, shell=True, timeout=60)

equil_steps = 10000
def analyze_freesolv():
    from solvation.sim import SolvationSim

    df = load_freesolv()
    with open("output/freesolv_results.txt", "w") as f:
        for i, row in df.iterrows():
            out_folder = os.path.join(CONFIG.cache_dir, f"freesolv_{equil_steps}_better", row.compound)
            os.makedirs(out_folder, exist_ok=True)


            print(f"Processing {row.compound_name}")
            print(f"Saving results to {out_folder}")

            lig_file = os.path.join(out_folder, "ligand.sdf")
            if not os.path.exists(lig_file):
                smi_to_protonated_sdf(row.smiles, lig_file)

            sim = SolvationSim(lig_file, out_folder)
            sim.equil_steps = equil_steps
            sim.run_all()
            delta_F = sim.compute_delta_F()

            print(f"Computed delta_F: {delta_F.value_in_unit(unit.kilocalories_per_mole)} kcal/mol")
            print(row)

            f.write(f"{row.compound_name},{delta_F.value_in_unit(unit.kilocalories_per_mole)},{row.exp_dG},{row.calc_dG}\n")
            f.flush()

if __name__ == "__main__":
    analyze_freesolv()



