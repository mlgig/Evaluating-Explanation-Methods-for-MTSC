from load_data import load_data
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import MiniRocketMultivariate, Rocket
from sklearn.linear_model import RidgeClassifierCV, LinearRegression
from resnet import Classifier_RESNET
import numpy as np
from utilities import convert2one_hot


def main():
    all_data = load_data("synth")

    # TODO take parameters for resent init from data

    for dataset in all_data.keys():
        print("dataset ",dataset)
        data = all_data[dataset]

        # temp main for getting accuracy using synth datasets
        acc_res,acc_mini,acc_MrFs, acc_MrClf = [],[],[],[]

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
            y_train,y_test,y_true = convert2one_hot(data["y_train"],data["y_test"])
            X_train, X_test = data["X_train"], data["X_test"]
            nTimes_nChannels = (data["X_train"].shape[-2] ,data["X_train"].shape[-1])
            resnet = Classifier_RESNET(input_shape=nTimes_nChannels,nb_classes=2)
            resnet.fit(X_train,y_train, X_test, y_test,nb_epochs=20)
            y_pred =  np.argmax( resnet.model(X_test), axis=1)
            acc_res.append((100 -np.sum(np.abs(data['y_test']-y_pred)) )/100)
        print("\t resnet accuracy is ",np.sum(acc_res)/5, acc_res)

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