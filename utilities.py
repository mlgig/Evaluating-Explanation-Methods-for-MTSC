import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import torch
from torch.utils.data import DataLoader

# TODO let see what I can eliminate/need to change here
def convert2one_hot(y_train,y_test):
    enc = OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)
    return y_train,y_test,y_true

def convert2tensor(data):
    if type(data)==np.ndarray:
        return  data
    else:
        dims = data.shape
        new_data = []
        for i in range(dims[0]):
            new_data.append([])
            for j in range(dims[1]):
                new_data[-1].append(data.values[i][j])
        return  tf.constant(new_data)
        #return tf.reshape( tf.constant(new_data),(dims[0],-1,dims[1]) )











def get_pytorch_datasets(data):
    encoder = OneHotEncoder(categories='auto', sparse=False)
    y_train = encoder.fit_transform(np.expand_dims(data['y_train'], axis=-1))
    y_val = encoder.transform(np.expand_dims(data['y_test'], axis=-1))

    device = device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train,= torch.from_numpy( data['X_train']).to(device)
    y_train = torch.Tensor(y_train).to(device)
    X_test, = torch.from_numpy(data['X_test']).to(device)
    y_test =  torch.Tensor(y_val).to(device)
    return DataLoader(training_data, batch_size=64, shuffle=True), DataLoader(test_data, batch_size=64, shuffle=True)