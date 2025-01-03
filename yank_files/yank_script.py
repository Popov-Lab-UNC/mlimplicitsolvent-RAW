import os
import subprocess
import pandas as pd
import yaml
import traceback
from simple_slurm import Slurm
import datetime
from tqdm import tqdm


def yank_script_generation(input, new_path, output_dir, mol_path):
    with open(input, 'r') as f:
        data = yaml.safe_load(f)
    data['options']['output_dir'] = output_dir
    data['molecules']['name']['filepath'] = mol_path

    with open(new_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_freesolv():
    return pd.read_csv(
        '/work/users/r/d/rdey/ml_implicit_solvent/freesolv/SAMPL.csv')


def smi_to_protonated_sdf(smiles, out_file):
    """ Uses openbabl to add a conformer + protonate the molecule, saving to out_file """
    cmd = f"obabel -:'{smiles.strip()}' -O '{out_file}' --gen3d --pH 7"
    subprocess.run(cmd, check=True, shell=True, timeout=60)


def make_all_files(df, master_path):
    master_file = r'yank_solv.yaml'

    for _, row in df.iterrows():
        try:
            name = row['iupac']
            smiles = row['smiles']
            expt = row['expt']
            calc = row['calc']

            curr = os.path.join(master_path, name)
            if (os.path.exists(curr)):
                print(f"{name} already exists.")
                continue
            os.mkdir(curr)
            mol_file = os.path.join(curr, f"{name}.mol2")
            yank_file = os.path.join(curr, f"{name}.yaml")

            smi_to_protonated_sdf(smiles=smiles, out_file=mol_file)
            yank_script_generation(input=master_file,
                                   new_path=yank_file,
                                   output_dir=curr,
                                   mol_path=mol_file)

        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()


def slurm_generation(master_path, name):
    slurm = Slurm(
        partition="l40-gpu",
        nodes=1,
        ntasks=1,
        job_name=f'yank_solv',
        time=datetime.timedelta(days=0, hours=1, minutes=0, seconds=0),
        mem="1g",
        qos="gpu_access",
        output=
        "/work/users/r/d/rdey/ml_implicit_solvent/yank_files/yank_output_slurmm/%j.out",
        gres="gpu:1")
    file = os.path.join(master_path, name, f"{name}.yaml")
    if (not os.path.exists(file)):
        return 1
    command = f"yank script --yaml='{file}'"
    slurm.add_cmd(f"source ~/.bashrc && mamba activate yank && {command}")
    slurm.sbatch()
    return 0


def run_all(master_path):
    df = load_freesolv()
    make_all_files(df, master_path)
    for _, row in df.iterrows():
        try:
            name = row['iupac']
            if (slurm_generation(master_path, name)):
                print("Failed on Slurmm")
        except:
            traceback.print_exc()


def run_analysis_all(master_path):
    df = load_freesolv()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        name = row['iupac']
        file_path = os.path.join(master_path, name)
        analysis_path = os.path.join(file_path, 'experiments')

        if not (os.path.exists(file_path) and os.path.exists(analysis_path)):
            tqdm.write(f"{name} file does not exist")
            continue

        output_file = os.path.join(file_path, f'{name}.out')
        command = f"yank analyze --store='{analysis_path}'"

        result = subprocess.run(command,
                                shell=True,
                                capture_output=True,
                                text=True)

        if result.returncode == 0:
            with open(output_file, "w") as file:
                file.write(result.stdout)
            tqdm.write(f"{name} analysis complete")
        else:
            tqdm.write(f"Error analyzing {name}: {result.stderr}")


def read_analysis(master_path, output_csv):

    df = load_freesolv()
    solvation_dict = {}
    for _, row in df.iterrows():
        name = row['iupac']
        file_path = os.path.join(master_path, name)
        output_file = os.path.join(file_path, f'{name}.out')
        if (os.path.exists(output_file)):
            with open(output_file, 'r') as file:
                line = file.readline()
                if "Free energy of solvation" in line:
                    solvation_dict[name] = float(line.split()[4])
                else:
                    print(f'{name} has an error')
        else:
            print(f'{name} output does not exist')

    solvation_df = pd.DataFrame(
        list(solvation_dict.items()),
        columns=['Molecule', 'Free Energy of Solvation'])
    solvation_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    master_path = '/work/users/r/d/rdey/all_yank'
    csv_path = '/work/users/r/d/rdey/ml_implicit_solvent/yank.csv'
    read_analysis(master_path, csv_path)
