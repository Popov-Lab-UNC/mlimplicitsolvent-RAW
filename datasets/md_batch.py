from typing import List
import h5py
import torch
import terrace as ter
from terrace.batch import _batch_repr


class MDData(ter.Batchable):
    """ Stores all the data we need for training
    the neural network """
    
    pos: torch.Tensor
    charges: torch.Tensor
    atomic_numbers: torch.Tensor
    forces: torch.Tensor

    lambda_sterics: torch.Tensor
    lambda_electrostatics: torch.Tensor
    sterics_derivative: torch.Tensor
    electrostatics_derivative: torch.Tensor
    atom_features: torch.Tensor
    
    @staticmethod
    def get_batch_type():
        return MDBatch

class MDBatch(ter.BatchBase[MDData]):
    """ A batch of MDData. This automatically generates
    the 'batch' tensor needed by TorchMD-Net. This is
    a tensor that contains the molecule index for each
    atom in the batch. """
    
    def __init__(self, items: List[MDData]):

        # concatenate all the tensors
        for key in items[0].__dict__.keys():
            first = getattr(items[0], key)
            if len(first.shape) == 0:
                # collate scalars like we would in normal batches
                collated = torch.stack([getattr(item, key) for item in items])
            else:
                collated = torch.cat([getattr(item, key) for item in items], 0)
            setattr(self, key, collated)

        # create batch tensor
        self.batch = torch.zeros(self.pos.shape[0], dtype=torch.long)
        cur_idx = 0
        for i, item in enumerate(items):
            self.batch[cur_idx:cur_idx+len(item.pos)] = i
            cur_idx += len(item.pos)

    def asdict(self):
        """ Convert to dict """
        return self.__dict__

    def __repr__(self):
        indent = "   "
        ret = f"MDBatch(\n"
        for key, val in self.asdict().items():
            val_str = _batch_repr(val).replace("\n", "\n" + indent)
            ret += indent +  f"{key}={val_str}\n"
        ret += ")"
        return ret

    def to(self, device):
        """ Move to device """
        ret = MDBatch.__new__(MDBatch)
        for key, val in self.asdict().items():
            setattr(ret, key, val.to(device))
        return ret