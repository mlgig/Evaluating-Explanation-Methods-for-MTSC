from load_data import load_data
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
import timeit
from utilities import *
from resnet_torch import ResNetBaseline

def main():
    all_data = load_data("CMJ")

    for dataset_name in all_data.keys():
        print("dataset ",dataset_name)
        data = all_data[dataset_name]

        # temp main for getting accuracy using synth datasets
        acc_res,acc_mini,acc_Mr_Fs_sax, acc_Mr_Clf_sax,acc_Mr_Fs_sfa, acc_Mr_Clf_sfa = [],[],[],[],[],[]
        n_run = 5

        # TODO try to increment batch size
        starttime = timeit.default_timer()
        for _ in range(n_run):
            test_dataloader, train_dataloader, n_channels, n_classes, device = transform_data4ResNet(data,dataset_name)
            model = ResNetBaseline(in_channels=n_channels, num_pred_classes=n_classes).double().to(device)
            acc = model.fit(train_dataloader, test_dataloader,num_epochs=20,patience=3)
            acc_res.append(acc)
        print("\t resnet accuracy is ",np.sum(acc_res)/n_run, acc_res," time was ", (timeit.default_timer() - starttime)/n_run)

        # TODO try np 3d array loader from https://www.sktime.org/en/v0.8.1/examples/loading_data.html#Using-NumPy-3d-arrays-with-sktime
        train_set = np.transpose(data["X_train"],(0,2,1)) if type(data["X_train"])==np.array else data["X_train"]
        test_set = np.transpose( data["X_test"],(0,2,1)) if type(data["X_test"])==np.array else data["X_test"]



        print(train_set.shape,test_set.shape)
        # TODO convert to a DataFrame for synth data
        starttime = timeit.default_timer()

        for _ in range(n_run):
            # rocket
            rocket = Rocket(normalise=False)

            parameters = rocket.fit(train_set)
            X_train_trans = rocket.transform(train_set, parameters)
            X_test_trans = rocket.transform(test_set,parameters)

            cls = RidgeClassifierCV()
            cls.fit(X_train_trans,data["y_train"])
            acc_mini.append( cls.score(X_test_trans,data["y_test"]) )
        print("\t rocket accuracy is ",np.sum(acc_mini)/n_run," time was ", (timeit.default_timer() - starttime)/n_run)


        starttime = timeit.default_timer()
        for _ in range(n_run):
            # TODO TRY sfa with MP
            #mrSeql
            model = MrSEQLClassifier(seql_mode="fs",symrep=['sax'])
            model.fit(train_set,data["y_train"])
            acc_Mr_Fs_sax.append( model.score(test_set,data["y_test"]) )
        print("\t mrSeql Fs sax accuracy is ", np.sum(acc_Mr_Fs_sax)/n_run," time was ",  (timeit.default_timer() - starttime)/n_run)

        starttime = timeit.default_timer()
        for _ in range(n_run):
            model = MrSEQLClassifier(seql_mode="clf",symrep=['sax'])
            model.fit(train_set,data["y_train"])
            acc_Mr_Clf_sax.append( model.score(test_set,data["y_test"]) )
        print("\t mrSeql Clf sax accuracy is ", np.sum(acc_Mr_Clf_sax)/n_run," time was ",  (timeit.default_timer() - starttime)/n_run)

        starttime = timeit.default_timer()
        for _ in range(n_run):
            starttime = timeit.default_timer()
            model = MrSEQLClassifier(seql_mode="fs",symrep=['sfa'])
            model.fit(train_set,data["y_train"])
            acc_Mr_Fs_sfa.append( model.score(test_set,data["y_test"]) )
        print("\t mrSeql Fs sfa accuracy is ", np.sum(acc_Mr_Fs_sfa)/n_run," time was ", (timeit.default_timer() - starttime)/n_run)



        starttime = timeit.default_timer()
        for _ in range(n_run):
            starttime = timeit.default_timer()
            model = MrSEQLClassifier(seql_mode="clf",symrep=['sfa'])
            model.fit(train_set,data["y_train"])
            acc_Mr_Fs_sfa.append( model.score(test_set,data["y_test"]) )
        print("\t mrSeql clf sfa accuracy is ", np.sum(acc_Mr_Clf_sfa)/n_run," time was ", (timeit.default_timer() - starttime)/n_run)
        


if __name__ == "__main__" :
    main()