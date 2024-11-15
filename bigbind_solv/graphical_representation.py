from epsilon_calculation import SolvDatasetReporterWithCustomDP, runSim
import os
import sys

def runAll(base_file_path, complete, curr_mod):
    curr = 1
    dp = 1e-5
    steps = 100000
    total = 1000 # until 1-(1e-4)
    for i in range(total):
        if(i % 5 == curr_mod):
            print(f"Current Lambda_Elec: {curr}")
            sim_path = os.path.join(complete, str(i))
            os.mkdir(sim_path)
            runSim(base_file_path, sim_path, steps, dp, curr, 1)
        curr -= dp



if __name__ == "__main__":
    base_file_path = "/work/users/r/d/rdey/BigBindDataset_New/521610"
    sim_path = "Graphical"
    complete = os.path.join(base_file_path, sim_path)
    if(not os.path.exists(complete)):
        os.mkdir(complete)
    curr_mod = int(sys.argv[1])
    runAll(base_file_path, complete, curr_mod)
