from simple_slurm import Slurm
import datetime


df800 = []
for i in range(0, 133885):
    if i%170 == 0:
        df800.append(i)
df800.append(133885)


'''
for i in range(0, 788):
    slurm = Slurm(
        partition = "general",
        nodes=1,
        ntasks=1,
        job_name='QM9_Solvation_Calculation',
        time=datetime.timedelta(days=5, hours=0, minutes=0, seconds=0),
        mem = "5g",
    )
    slurm.add_cmd('source ~/.bashrc')
    slurm.add_cmd('mamba activate implicit')
    command = f'python qm9_calc.py {df800[i]} {df800[i+1]} {i}'
    print(command)
    slurm.sbatch(command)
    print(f"Ran SLURM {i} from {df800[i]}-{df800[i+1]}")

'''

slurm = Slurm(
        partition = "general",
        nodes=1,
        ntasks=1,
        job_name='QM9_Solvation_Calculation',
        time=datetime.timedelta(days=1, hours=0, minutes=0, seconds=0),
        mem = "10g",
    )
command = f'python qm9_calc.py 41480 41650 244'
slurm.add_cmd(f'source ~/.bashrc && conda activate implicit && {command}')
slurm.sbatch()
