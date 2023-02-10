from load_data import load_data
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
import timeit
from utilities import *
#from resnet_torch import ResNetBaseline

#TODO is there a better way to import the files?
import sys
base_path="./dCAM/src/"
sys.path.insert(0, base_path+'explanation')
sys.path.insert(0, base_path+'models')
from DCAM import *
from CNN_models import *

def main():
    all_data = load_data("MP")

    for dataset_name in all_data.keys():
        print("dataset ",dataset_name)
        data = all_data[dataset_name]

        # temp main for getting accuracy using synth datasets
        n_run = 5

        acc_res = []
        starttime = timeit.default_timer()
        for i in range(n_run):
            train_dataloader,test_dataloader, n_channels, n_classes, device = transform_data4ResNet(data,dataset_name)
            # TODO check carefully data types in the hole process as label types?
            # TODO also check the conversion as .float() etc. Can I avoid some of them?
            modelarch = dResNetBaseline(n_channels,mid_channels=64,num_pred_classes=n_classes).to(device)
            model = ModelCNN(model=modelarch ,n_epochs_stop=30,device=device)# ,save_path='saved_model/resNet/'+dataset_name+str(i))
            acc = model.train(num_epochs=1000,train_loader=train_dataloader,test_loader=test_dataloader)
            acc_res.append(acc)
        print("\t resnet accuracy is ",np.sum(acc_res)/n_run, acc_res," time was ", (timeit.default_timer() - starttime)/n_run)

        #TODO move above train and test pointers
        train_set = data["X_train"]
        test_set =  data["X_test"]


        print(train_set.shape,test_set.shape)
        # TODO convert to a DataFrame for synth data
        starttime = timeit.default_timer()

        for normal in [True,False]:
            acc_mini=[]
            for _ in range(n_run):

                # rocket
                rocket = Rocket(normalise=normal)

                parameters = rocket.fit(train_set)
                X_train_trans = rocket.transform(train_set, parameters)
                X_test_trans = rocket.transform(test_set,parameters)

                cls = LogisticRegressionCV(solver='newton-cg',multi_class = 'multinomial', class_weight='balanced')
                cls.fit(X_train_trans,data["y_train"])
                acc = cls.score(X_test_trans,data["y_test"])
                acc_mini.append(acc)
            print("\t rocket normal ",normal," accuracy is ",np.sum(acc_mini)/n_run," time was ", (timeit.default_timer() - starttime)/n_run)


        starttime = timeit.default_timer()
        for seql in ["fs","clf"]:
            acc_mrSEQL=[]
            for _ in range(n_run):
                model = MrSEQLClassifier(seql_mode=seql,symrep=['sax'])
                model.fit(train_set,data["y_train"])
                acc_mrSEQL.append( model.score(test_set,data["y_test"]) )
            print("\t mrSeql ",seql," sax accuracy is ", np.sum(acc_mrSEQL)/n_run," time was ",  (timeit.default_timer() - starttime)/n_run)




if __name__ == "__main__" :
    main()