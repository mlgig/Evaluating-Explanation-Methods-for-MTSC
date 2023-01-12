from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from utilities import convert2tensor
import pandas as pd
import numpy as np
import torch

"""
class myDataset(Dataset):
    def __init__(self, X,y, device):
        if type(X)==np.ndarray:
            self.X= torch.from_numpy( np.transpose(X,(0,2,1)) ).to(device)
        elif type(X)==pd.DataFrame:
            self.X = convert2tensor(X)
        else:
            raise ValueError("dataset format not recognized")
        self.y= torch.Tensor(y).to(device)
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    """