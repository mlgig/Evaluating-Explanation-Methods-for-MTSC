from load_data import load_data
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.linear_model import RidgeClassifierCV, LinearRegression
import numpy as np
import sklearn
from resnet_tensorflow import Classifier_RESNET
import time


def convert2one_hot(y_train,y_test):
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)
    return y_train,y_test,y_true

def main():
    all_data = load_data("synth")

    out_dir = "/home/davide/workspace/PhD/Trang/first_experiment/resNet_out"
    #TODO take parameters for resent init from data

    for dataset in all_data.keys():
        print("dataset ",dataset)
        data = all_data[dataset]

        #resnet
        resnet = Classifier_RESNET(out_dir,(100,20),2)
        y_train,y_test,y_true = convert2one_hot(data["y_train"],data["y_test"])
        X_train, X_test = data["X_train"], data["X_test"]
        resnet.fit(X_train,y_train, X_test, y_test, y_true,10)

        #minirocket
        start = time.time()
        minirocket = MiniRocketMultivariate()

        parameters = minirocket.fit(data["X_train"])
        X_train_trans = minirocket.transform(data["X_train"], parameters)
        X_test_trans = minirocket.transform(data["X_test"],parameters)

        cls = RidgeClassifierCV()
        cls.fit(X_train_trans,data["y_train"])
        acc = cls.score(X_test_trans,data["y_test"])
        print("\t minirocket accuracy is ",acc, " in ", time.time()-start, " seconds")

        #mrSeql
        start = time.time()
        model = MrSEQLClassifier(seql_mode="fs")
        model.fit(data["X_train"],data["y_train"])
        acc = model.score(data["X_test"],data["y_test"])
        print("\t mrSeql accuracy is ",acc, " in ", time.time()-start, " seconds")



if __name__ == "__main__" :
    main()