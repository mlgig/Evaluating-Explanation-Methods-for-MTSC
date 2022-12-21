import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

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

        return tf.reshape( tf.constant(new_data),(dims[0],-1,dims[1]) )