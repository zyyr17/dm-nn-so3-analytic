import torch
import torch.nn as nn
import numpy as np
import itertools
from .sqlite_database import HamiltonianDatabase

dtype1 = torch.float64

class HamiltonianDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, dtype=dtype1):
        super(HamiltonianDataset, self).__init__()
        self.dtype  = dtype
        self.database = HamiltonianDatabase(filepath)

        #collect the orbitals, which give the shape of the hamiltonian
        orbitals = []
        for Z in self.database.Z:
            orbitals.append(tuple((int(Z),int(l)) for l in self.database.get_orbitals(Z)))
        self.orbitals = tuple(orbitals)

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        return idx #just return the idx, the custom collate_fn does the querying

    #collate function to collate several data points into a batch, should be passed
    #to data loader (default_collate is not efficient, because there should be a batch-wise query)
    def collate_fn(self, batch):
        all_data = self.database[batch] #fetch the batch data
        R, E, H, P, S, C = [], [], [], [], [], []
        for batch_num, data in enumerate(all_data):
            R_, E_, H_, P_, S_, C_ = data
            R.append(torch.tensor(R_, dtype=self.dtype))
            E.append(torch.tensor(E_, dtype=self.dtype))
            H.append(torch.tensor(H_, dtype=self.dtype))
            P.append(torch.tensor(P_, dtype=self.dtype))
            S.append(torch.tensor(S_, dtype=self.dtype))
            C.append(torch.tensor(C_, dtype=self.dtype))

        return {'positions': torch.stack(R), 
                'energy': torch.stack(E),
                'H_matrix': torch.stack(H),
                'density_matrix': torch.stack(P),
                'overlap_matrix': torch.stack(S),
                'core_hamiltonian': torch.stack(C)}     

def seeded_random_split(dataset, lengths, seed=None):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    This is very similar to the pytorch equivalent, but this version allows a seed to
    be specified in order to make the split reproducible

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = np.random.RandomState(seed=seed).permutation(sum(lengths))
    return [torch.utils.data.Subset(dataset, indices[offset - length:offset]) for offset, length in zip(itertools.accumulate(lengths), lengths)]