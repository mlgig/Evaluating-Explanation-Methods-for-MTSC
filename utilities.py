from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from scipy.interpolate import interp1d
from dCAM.src.models.CNN_models import TSDataset


def convert2tensor(data):
    dims = data.shape
    new_data = []
    for i in range(dims[0]):
        tmp = []
        for j in range(dims[1]):
            tmp.append(data.values[i][j])
        new_data.append(tmp)

    return  torch.tensor(new_data, dtype=torch.float64)


def one_hot_encoding(train_labels,test_labels):
    enc = LabelEncoder()
    y_train = enc.fit_transform(train_labels)
    y_test = enc.transform(test_labels)

    return y_train,y_test,enc

def transform_data4ResNet(data,dataset_name,concat):

    # get dataset loaders
    device = device = "cuda" if torch.cuda.is_available() else "cpu"
    n_channels = data["X_train"].shape[1] if not concat else 1
    n_classes = len( np.unique(data["y_train"]) )

    if type(data['X_train']) == pd.DataFrame:
        train_set_cube = np.array([gen_cube(acl) for acl in data["X_train"].values.tolist()])
        test_set_cube = np.array([gen_cube(acl) for acl in data["X_test"].values.tolist()])
    else:
        if len(data["X_train"].shape)==3:
            train_set_cube = np.array([gen_cube(acl) for acl in data["X_train"].tolist()])
            test_set_cube = np.array([gen_cube(acl) for acl in data["X_test"].tolist()])
        elif len(data["X_train"].shape)==2:
            train_set_cube = np.array([gen_cube([acl]) for acl in data["X_train"].tolist()])
            test_set_cube = np.array([gen_cube([acl]) for acl in data["X_test"].tolist()])

    if dataset_name=="CMJ":
        batch_s = (32,32)
    elif dataset_name=="MP":
        batch_s = (64,64)
    else:
        batch_s = (32,32)

    y_train,y_test,enc = one_hot_encoding( data["y_train"],data["y_test"] )

    #TODO fix for MP. better to use the one to hot before?
    train_loader = DataLoader(TSDataset(train_set_cube,y_train), batch_size=batch_s[0],shuffle=True)
    test_loader = DataLoader(TSDataset(test_set_cube,y_test), batch_size=batch_s[1],shuffle=False)

    return train_loader, test_loader,n_channels,n_classes, device, enc

def interpolation(x,max_length,n_var):
    n = len(x)
    interpolated_data = np.zeros((n, n_var,max_length), dtype=np.float64)

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
            interpolated_data[i, j, :] = new_ts
    return interpolated_data


def gen_cube(instance):
    result = []
    for i in range(len(instance)):
        result.append([instance[(i+j)%len(instance)] for j in range(len(instance))])
    return result