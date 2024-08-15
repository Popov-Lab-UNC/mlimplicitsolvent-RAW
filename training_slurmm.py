from simple_slurm import Slurm
import datetime
slurm = Slurm(job_name = "GNNRun",
              partition = 'l40-gpu',
              gres = 'gpu:1',
              qos = 'gpu_access', 
              mem = "16g", 
              time=datetime.timedelta(days=1, hours=2, minutes=3, seconds=4), 
              cpus_per_task=1, 
              )

command = "python -m train"
slurm.sbatch(f"source ~/.bashrc && module load cuda && mamba activate GNNProject && {command}")

