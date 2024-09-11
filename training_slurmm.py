from simple_slurm import Slurm
import datetime
slurm = Slurm(job_name = "GNNRun",
              partition = 'l40-gpu',
              gres = 'gpu:1',
              qos = 'gpu_access', 
              mem = "16g", 
              time=datetime.timedelta(days=0, hours=1, minutes=0, seconds=4), 
              cpus_per_task=1, 
              output = "/work/users/r/d/rdey/ml_implicit_solvent/slurm_output/%j.out"
              )

command = "python aisim.py"
slurm.sbatch(f"source ~/.bashrc && module load cuda && mamba activate GNNProject && {command}")

