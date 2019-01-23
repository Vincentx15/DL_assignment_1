from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class ToyData(Dataset):
    def __init__(self):
        df = pd.read_csv('data/toy.csv', dtype=np.float32, index_col=0)
        self.data = df.values

    def __getitem__(self, id):
        row = self.data[id]
        return row[:2], row[2]

    def __len__(self):
        return len(self.data)
