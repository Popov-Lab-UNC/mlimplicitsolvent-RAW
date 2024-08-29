from simple_slurm import Slurm
import datetime
slurm = Slurm(job_name = "GNNRun",
              partition = 'l40-gpu',
              gres = 'gpu:1',
              qos = 'gpu_access', 
              mem = "16g", 
              time=datetime.timedelta(days=0, hours=0, minutes=40, seconds=4), 
              cpus_per_task=4, 
              output = "/work/users/r/d/rdey/ml_implicit_solvent/slurm_output/%j.out"
              )

command = "python -m train"
slurm.sbatch(f"source ~/.bashrc && module load cuda && mamba activate GNNProject && {command}")

