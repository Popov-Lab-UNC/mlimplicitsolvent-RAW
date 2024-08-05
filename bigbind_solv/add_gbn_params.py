from typing import List
import h5py
import os
from tqdm import trange
from config import CONFIG
from datasets.bigbind_solv import BigBindSolvDataset
from openmm.app.internal.customgbforces import GBSAGBn2Force
from datasets.md_batch import *
from bigbind_solv.gb_baseline import *
from traceback import print_exc
import signal


class timeout:
    def __init__(self, seconds, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def add_gbn_params():
    os.makedirs(CONFIG.bigbind_solv_dir, exist_ok=True)

    for split in ["val", "test", "train"]:
        dataset = BigBindSolvDataset(split, 0, dir=CONFIG.bigbind_solv_dir_old)
        h5_fname = os.path.join(CONFIG.bigbind_solv_dir, f"{split}.h5")
        if os.path.exists(h5_fname):
            os.remove(h5_fname)

        h5_file = h5py.File(h5_fname, "w")

        print("Adding Gbn parameters to", split)
        for index in trange(len(dataset)):
            try:
                with timeout(5):
                    key = dataset.keys[index]

                    data = dataset[index]
                    topology = to_openmm_topology(data)
                    force = GBSAGBn2Force(cutoff=None,SA="ACE")
                    gbn2_params = force.getStandardParameters(topology).shape

                    out_group = h5_file.create_group(key)
                    in_group = dataset.file[key]
                    for k, v in in_group.items():
                        out_group[k] = v[()]
                    out_group["gbn2_params"] = gbn2_params
            except KeyboardInterrupt:
                raise
            except:
                print_exc()
        
        h5_file.close()

if __name__ == "__main__":
    add_gbn_params()
