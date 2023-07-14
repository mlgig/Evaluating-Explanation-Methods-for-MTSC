from load_data import load_data
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from  sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import timeit
from utilities import *
from joblib import dump

# importing srcs for dResNet and dCAM
import sys
base_path="./dCAM/src/"
sys.path.insert(0, base_path+'explanation')
sys.path.insert(0, base_path+'models')
from DCAM import *
from CNN_models import *

def main():
    concat = False
    all_data = load_data("MP",concat=concat)
    n_run = 5

    for dataset_name in all_data.keys():
        # train ridge Classifier
        print("dataset ",dataset_name)
        data = all_data[dataset_name]
        train_set = data["X_train"]
        test_set =  data["X_test"]
        print(train_set.shape)

        if concat:
            # Ridge can be used only with concatenated datasets
            cls = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),cv=5)
            cls.fit(train_set, data["y_train"])
            acc = cls.score(test_set,data["y_test"])
            print(dataset_name,acc)
            dump(cls, "saved_model/Ridge/"+dataset_name)

        # train dResNet>
        acc_res = []
        for mid_channels in [64,128]:
            # train both 64 and 128 channels
            starttime = timeit.default_timer()
            for i in range(n_run):
                train_dataloader,test_dataloader, n_channels, n_classes, device,_ = transform_data4ResNet(
                    data,dataset_name,concat=concat)
                modelarch = dResNetBaseline(n_channels,mid_channels=mid_channels,num_pred_classes=n_classes).to(device)

                model = ModelCNN(model=modelarch ,n_epochs_stop=100,device=device,save_path='saved_model/resNet/'
                                                            +dataset_name+"_nFilters_"+str(mid_channels)+"_"+str(i)  )
                acc = model.train(num_epochs=11,train_loader=train_dataloader,test_loader=test_dataloader)
                acc_res.append(acc)
            print("\t resnet accuracy is ",np.sum(acc_res)/n_run," time was ", (timeit.default_timer() - starttime)/n_run)

        # train Rocket
        for normal in [False,True]:
            # if concat dataset expand dim
            if concat:
                train_set = np.expand_dims(train_set,1)
                test_set = np.expand_dims(test_set,1)

            starttime = timeit.default_timer()
            acc_rocket=[]
            for i in range(0,n_run):
                # rocket
                cls = make_pipeline(Rocket(normalise=normal,n_jobs=-1) ,StandardScaler(),
                                    LogisticRegressionCV(cv = 5, random_state=0, n_jobs = -1,max_iter=1000))
                cls.fit(train_set,data["y_train"])
                acc = cls.score(test_set,data["y_test"])
                acc_rocket.append(acc)
                dump(cls,"saved_model/rocket/"+dataset_name+"_new_norm_"+str(normal)+"_"+str(i))
            print("\t rocket normal ",normal," accuracy is ",np.sum(acc_rocket)/n_run," time was ", (timeit.default_timer() - starttime)/n_run)

if __name__ == "__main__" :
    main()