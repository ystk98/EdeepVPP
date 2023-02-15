import math
import numpy as np
import torch
from  torch.utils.data import DataLoader

"""
General: 
    DataLoader class returns datum of Dataset as mini-batched Iterator.
    In addition, DataLoader shuffles the data order.

"""

# =============================================================================
# BatchedGenomeDataLoader
# =============================================================================
"""
Note:
    Input: fixed-length genome data
    Output: mini-batched Iterator.

"""

class BatchedGenomeDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1000, shuffle=True):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size/batch_size) # 小数点以下切り上げ

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration
        
        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i*batch_size:(i + 1)*batch_size]
        batch = [self.dataset[i] for i in batch_index]
        x = torch.from_numpy(np.array([example[0] for example in batch])).cuda().to(torch.float32)
        t = torch.tensor([example[1] for example in batch], dtype=torch.float32).cuda()

        self.iteration += 1
        
        return x, t

    def next(self):
        return self.__next__()