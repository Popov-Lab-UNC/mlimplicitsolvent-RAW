from typing import List
import h5py
import os
from tqdm import trange
from datasets.bigbind_solv import MAFBigBind
from openmm.app.internal.customgbforces import GBSAGBn2Force
from datasets.md_batch import *
from bigbind_solv.gb_baseline import *
from traceback import print_exc
import multiprocessing as mp


def make_openmm_topology(data, ret_queue):
    try:
        ret_queue.put(to_openmm_topology(data))
    except KeyboardInterrupt:
        raise
    except:
        print_exc()
        ret_queue.put(None)


def add_gbn_params():
    os.makedirs('/work/users/r/d/rdey/BigBindDataset_New/bigbind_solv',
                exist_ok=True)

    for split in ["val", "test", "train"]:
        dataset = MAFBigBind(
            split, dir='/work/users/r/d/rdey/BigBindDataset_New/bigbind_solv')
        h5_gnn_fname = os.path.join(
            '/work/users/r/d/rdey/BigBindDataset_New/bigbind_solv',
            f"{split}_GNN.h5")
        if os.path.exists(h5_fname):
            os.remove(h5_fname)

        h5_file = h5py.File(h5_fname, "w")

        print("Adding Gbn parameters to", split)
        for index in trange(len(dataset)):
            try:
                key = dataset.keys[index]

                data = dataset[index]

                # annoying way to make sure this times out after 1 sec
                q = mp.Queue()
                p = mp.Process(target=make_openmm_topology, args=(data, q))
                p.start()
                p.join(5)
                if p.is_alive():
                    p.terminate()
                    p.join()
                    print("Timeout for index", index)
                    continue

                topology = q.get()
                if topology is None:
                    print("Error at index", index)
                    continue

                force = GBSAGBn2Force(cutoff=None, SA="ACE")
                gbn2_params = force.getStandardParameters(topology)

                out_group = h5_file.create_group(key)
                in_group = dataset.file[key]
                for k, v in in_group.items():
                    out_group[k] = v[()]
                out_group["gbn2_params"] = gbn2_params
            except KeyboardInterrupt:
                raise
            except:
                print("Error at index", index)
                print_exc()

        h5_file.close()


if __name__ == "__main__":
    add_gbn_params()
