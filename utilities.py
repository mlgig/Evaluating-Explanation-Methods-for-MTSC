from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import torch


def convert2tensor(data):
    dims = data.shape
    new_data = []
    for i in range(dims[0]):
        tmp = []
        for j in range(dims[1]):
            tmp.append(data.values[i][j])
        new_data.append(tmp)

    return  torch.tensor(new_data, dtype=torch.float64)

class myDataset(Dataset):
    def __init__(self, X,y, device):
        if type(X)==np.ndarray:
            self.X= torch.from_numpy( np.transpose(X,(0,2,1)) ).to(device)
        elif type(X)==pd.DataFrame:
            self.X = convert2tensor(X).to(device)
        else:
            raise ValueError("dataset format not recognized")
        self.y= torch.Tensor(y).to(device)
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_n_channels(self):
        return self.X.shape[1]

def transform_data4ReseNet(data):
    # get one hot encoder
    encoder = OneHotEncoder(categories='auto', sparse=False)
    y_train = encoder.fit_transform(np.expand_dims(data["y_train"], axis=-1))
    y_test = encoder.transform(np.expand_dims(data["y_test"], axis=-1))

    # get dataset loaders
    device = device = "cuda" if torch.cuda.is_available() else "cpu"
    train = myDataset(data["X_train"], y_train, device)
    test =  myDataset(data["X_test"], y_test, device)
    train_dataloader = DataLoader(train, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=100, shuffle=True)

    # get how many classes and channels
    n_channels = train.get_n_channels()
    assert train.get_n_channels()==test.get_n_channels()
    n_classes =  encoder.categories_[0].size

    return test_dataloader, train_dataloader,n_channels,n_classes, device
