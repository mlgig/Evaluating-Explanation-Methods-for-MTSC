from load_data import load_data
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import MiniRocketMultivariate, Rocket
from sklearn.linear_model import RidgeClassifierCV, LinearRegression
import numpy as np
import sklearn
from resnet import Classifier_RESNET


# TODO move in a new helper file
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
    # TODO take parameters for resent init from data

    for dataset in all_data.keys():
        print("dataset ",dataset)
        data = all_data[dataset]

        #resnet
        #TODO number of channels taken from dataset rather than costant

        acc_res,acc_mini,acc_MrFs, acc_MrClf = [],[],[],[]

        for _ in range(5):
            resnet = Classifier_RESNET(out_dir,(100,20),2,build=True)
            y_train,y_test,y_true = convert2one_hot(data["y_train"],data["y_test"])
            X_train, X_test = data["X_train"], data["X_test"]
            resnet.fit(X_train,y_train, X_test, y_test, y_true,20)
            y_pred =  np.argmax( resnet.model(X_test), axis=1)
            acc_res.append((100 -np.sum(np.abs(data['y_test']-y_pred)) )/100)
        print("\t resnet accuracy is ",np.sum(acc_res)/5, acc_res)
        exit()

        for _ in range(5):
            # rocket
            rocket = Rocket(normalise=False)

            parameters = rocket.fit(data["X_train"])
            X_train_trans = rocket.transform(data["X_train"], parameters)
            X_test_trans = rocket.transform(data["X_test"],parameters)

            cls = RidgeClassifierCV()
            cls.fit(X_train_trans,data["y_train"])
            acc_mini.append( cls.score(X_test_trans,data["y_test"]) )
        print("\t rocket accuracy is ",np.sum(acc_mini)/5)

        for _ in range(5):
            #mrSeql
            model = MrSEQLClassifier(seql_mode="fs",symrep=['sfa'])
            model.fit(data["X_train"],data["y_train"])
            acc_MrFs.append( model.score(data["X_test"],data["y_test"]) )
        print("\t mrSeqlFs accuracy is ", np.sum(acc_MrFs)/5)

        for _ in range(5):
            #mrSeql
            model = MrSEQLClassifier(seql_mode="clf",symrep=['sfa'])
            model.fit(data["X_train"],data["y_train"])
            acc_MrClf.append( model.score(data["X_test"],data["y_test"]) )
        print("\t mrSeqlClf accuracy is ", np.sum(acc_MrClf)/5)



if __name__ == "__main__" :
    main()