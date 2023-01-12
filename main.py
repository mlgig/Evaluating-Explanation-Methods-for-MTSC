from load_data import load_data
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
import numpy as np
from resnet_torch import  ResNetBaseline
from utilities import *
from torch.utils.data import DataLoader
import torch
from sklearn.preprocessing import OneHotEncoder


def main():
    all_data = load_data("MP")


    for dataset in all_data.keys():
        print("dataset ",dataset)
        data = all_data[dataset]

        # temp main for getting accuracy using synth datasets
        acc_res,acc_mini,acc_MrFs, acc_MrClf = [],[],[],[]
        n_run = 1


        for _ in range(n_run):
            test_dataloader, train_dataloader, n_channels, n_classes, device = transform_data4ReseNet(data)
            model = ResNetBaseline(in_channels=n_channels, num_pred_classes=n_classes).double().to(device)
            model.fit(train_dataloader, test_dataloader,num_epochs=20)
        ##print("\t resnet accuracy is ",np.sum(acc_mini)/n_run)


        #TODO convert to a DataFrame for synth data
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
            # TODO TRY sfa with MP
            #mrSeql
            model = MrSEQLClassifier(seql_mode="fs",symrep=['sax'])
            model.fit(data["X_train"],data["y_train"])
            acc_MrFs.append( model.score(data["X_test"],data["y_test"]) )
        print("\t mrSeqlFs accuracy is ", np.sum(acc_MrFs)/n_run)

        for _ in range(n_run):
            #mrSeql
            model = MrSEQLClassifier(seql_mode="clf",symrep=['sax'])
            model.fit(data["X_train"],data["y_train"])
            acc_MrClf.append( model.score(data["X_test"],data["y_test"]) )
        print("\t mrSeqlClf accuracy is ", np.sum(acc_MrClf)/n_run,"\n\n\n\n")



if __name__ == "__main__" :
    main()