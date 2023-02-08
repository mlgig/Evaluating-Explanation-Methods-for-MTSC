from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
from scipy.interpolate import interp1d
from dCAM.src.models.CNN_models import TSDataset


labels = ['cat', 'dog', 'mouse', 'elephant', 'pandas']
le = preprocessing.LabelEncoder()
targets = le.fit_transform(labels)

def convert2tensor(data):
    dims = data.shape
    new_data = []
    for i in range(dims[0]):
        tmp = []
        for j in range(dims[1]):
            tmp.append(data.values[i][j])
        new_data.append(tmp)

    return  torch.tensor(new_data, dtype=torch.float64)

"""
class myDataset(Dataset):
    def __init__(self, X,y, device):
        if type(X)==np.ndarray:
            self.X= torch.from_numpy( np.transpose(X,(0,2,1)) ).to(device)
        elif type(X)==pd.DataFrame:
            self.X = convert2tensor(X).to(device)
        else:
            raise ValueError("dataset input format not recognized")

        if y.dtype==np.dtype("U4") or y.dtype==np.dtype("U1"):
            # that is the MP dataset
            le = preprocessing.LabelEncoder()
            targets = le.fit_transform(y)
            self.y= torch.tensor(targets, dtype=torch.uint8).to(device)
        elif y.dtype==np.int64:
            self.y= torch.tensor(y, dtype=torch.uint8).to(device)
        #elif:
        #    a = 3
            #TODO IMPLEMENT
        else:
            raise ValueError("dataset output format not recognized")
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_n_channels(self):
        return self.X.shape[1]

def transform_data4ResNet(data,dataset_name):

    # get dataset loaders
    device = device = "cuda" if torch.cuda.is_available() else "cpu"
    train = myDataset(data["X_train"], data["y_train"], device)
    test =  myDataset(data["X_test"], data["y_test"], device)
    train_dataloader = DataLoader(train, batch_size=16, shuffle=True) if dataset_name=="CMJ" else DataLoader(train, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=64, shuffle=False)

    # get how many classes and channels
    n_channels = train.get_n_channels()
    assert train.get_n_channels()==test.get_n_channels()
    n_classes = np.unique(data["y_train"]).size

    return test_dataloader, train_dataloader,n_channels,n_classes, device
"""
def transform_data4ResNet(data,dataset_name):

    # get dataset loaders
    device = device = "cuda" if torch.cuda.is_available() else "cpu"
    n_channels = data["X_train"].shape[1]
    n_classes = len( np.unique(data["y_train"]) )

    train_set_cube = np.array([gen_cube(acl) for acl in data["X_train"].tolist()])
    test_set_cube = np.array([gen_cube(acl) for acl in data["X_train"].tolist()])

    batch_s = (16,8) if dataset_name=="CMJ" else (32,32)
    train_loader = DataLoader(TSDataset(train_set_cube,data["y_train"]), batch_size=batch_s[0],shuffle=True)
    test_loader = DataLoader(TSDataset(test_set_cube,data["y_test"]), batch_size=batch_s[1],shuffle=True)

    return train_loader, test_loader,n_channels,n_classes, device

def interpolation(x,max_length,n_var):
    n = len(x)
    interpolated_data = np.zeros((n, max_length, n_var), dtype=np.float64)

    for i in range(n):
        mts = x[i]
        curr_length = mts[0].size
        idx = np.array(range(curr_length))
        idx_new = np.linspace(0, idx.max(), max_length)
        # TODO count for the pids
        # pid = mts[0][-1]
        for j in range(n_var):
            ts = mts[j]
            # linear interpolation
            f = interp1d(idx, ts, kind='cubic')
            new_ts = f(idx_new)
            interpolated_data[i, :, j] = new_ts
        # interpolated_data[i, :, -1] = pid
    return interpolated_data

#TODO add quote
def gen_cube(instance):
    result = []
    for i in range(len(instance)):
        result.append([instance[(i+j)%len(instance)] for j in range(len(instance))])
    return result