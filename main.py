from load_data import load_data
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from resnet import Classifier_RESNET
import numpy as np
from utilities import convert2one_hot, convert2tensor
from resnet_torch import  ResNetBaseline

def main():
    all_data = load_data("synth")

    for dataset in all_data.keys():
        print("dataset ",dataset)
        data = all_data[dataset]

        # temp main for getting accuracy using synth datasets
        acc_res,acc_mini,acc_MrFs, acc_MrClf = [],[],[],[]

        n_run = 1



        #TODO improve cuda, nb_classes from data
        device = "cuda"
        nb_classes = 2
        model = ResNetBaseline(in_channels=100, num_pred_classes=nb_classes).to(device)
        model.fit(data)





        exit()
        for _ in range(n_run):
            y_train,y_test,y_true = convert2one_hot(data["y_train"],data["y_test"])
            X_train, X_test = convert2tensor(data["X_train"]), convert2tensor( data["X_test"])
            nTimes_nChannels = (X_train.shape[-2] ,X_train.shape[-1])
            resnet = Classifier_RESNET(input_shape=nTimes_nChannels,nb_classes=y_train.shape[-1])
            hist = resnet.fit(X_train,y_train, X_test, y_test,nb_epochs=20)
            acc_res.append(np.max(hist.history["val_accuracy"]))
        print("\t resnet accuracy is ",np.sum(acc_res)/n_run, acc_res)



        for _ in range(n_run):
            # rocket
            rocket = Rocket(normalise=False)

            parameters = rocket.fit(data["X_train"])
            X_train_trans = rocket.transform(data["X_train"], parameters)
            X_test_trans = rocket.transform(data["X_test"],parameters)

            cls = RidgeClassifierCV()
            cls.fit(X_train_trans,data["y_train"])
            acc_mini.append( cls.score(X_test_trans,data["y_test"]) )
        print("\t rocket accuracy is ",np.sum(acc_mini)/n_run)




        for _ in range(n_run):
            #mrSeql
            model = MrSEQLClassifier(seql_mode="fs",symrep=['sfa'])
            model.fit(data["X_train"],data["y_train"])
            acc_MrFs.append( model.score(data["X_test"],data["y_test"]) )
        print("\t mrSeqlFs accuracy is ", np.sum(acc_MrFs)/n_run)

        for _ in range(n_run):
            #mrSeql
            model = MrSEQLClassifier(seql_mode="clf",symrep=['sfa'])
            model.fit(data["X_train"],data["y_train"])
            acc_MrClf.append( model.score(data["X_test"],data["y_test"]) )
        print("\t mrSeqlClf accuracy is ", np.sum(acc_MrClf)/n_run)


if __name__ == "__main__" :
    main()