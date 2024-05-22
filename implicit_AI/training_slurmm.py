from simple_slurm import Slurm
import datetime

slurm = Slurm(
        partition = "volta-gpu",
        nodes=1,
        ntasks=1,
        job_name='Training_QM9_IS',
        time=datetime.timedelta(days=5, hours=0, minutes=0, seconds=0),
        mem = "16g",
        qos='gpu_access',
        gres = 'gpu:1'
    )

command = f'python training.py'
slurm.add_cmd(f'source ~/.bashrc && module load cuda && mamba activate ml_implicit && {command}')
slurm.sbatch()
