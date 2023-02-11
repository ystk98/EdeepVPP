import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from functions import integer_encoding, label_encoding


# =============================================================================
# GenomeDataset / GigGenomeDataset
# =============================================================================
class GenomeDataset(Dataset):
    """
    Note: Base Dataset

        Input: 
            data_path (string): csv file data path
        
    """
    def __init__(self, data_path):
        self.data = None
        self.label = None
        self.data_path = data_path
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.data[index], None
        else:
            return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass


class BigGenomeDataset(GenomeDataset):
    """
    Note: BigGenomeDataset can be used for genome datasets which are too big to be expanded to computer memory.

        Input: 
            data_path (string): csv file data path
        
    """
    def __init__(self, data_path):
        super().__init__(data_path)
        self.df = pd.read_csv(data_path, header=None)

    def __getitem__(self, index):
        x = integer_encoding(self.df[1][index])
        t = label_encoding(self.df[2][index])

        return x, t

    def __len__(self):
        return len(self.df)