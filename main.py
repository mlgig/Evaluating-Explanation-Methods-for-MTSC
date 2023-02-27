from load_data import load_data
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from  sklearn.preprocessing import StandardScaler
import timeit
from utilities import *
from sklearn.pipeline import make_pipeline
from joblib import dump, load
#from resnet_torch import ResNetBaseline



#import warnings
#warnings.filterwarnings("ignore")

#TODO is there a better way to import the files?
import sys
base_path="./dCAM/src/"
sys.path.insert(0, base_path+'explanation')
sys.path.insert(0, base_path+'models')
from DCAM import *
from CNN_models import *

def main():
    all_data = load_data("CMJ")

    for dataset_name in all_data.keys():
        print("dataset ",dataset_name)
        data = all_data[dataset_name]
        train_set = data["X_train"]
        test_set =  data["X_test"]

        # temp main for getting accuracy using synth datasets
        n_run = 1

        """
        acc_res = []
        starttime = timeit.default_timer()
        mid_channels=64 if (dataset_name=="MP" or dataset_name=="CMJ")  else 128
        print("filters",mid_channels)
        for i in range(n_run):
            train_dataloader,test_dataloader, n_channels, n_classes, device = transform_data4ResNet(data,dataset_name)
            modelarch = dResNetBaseline(n_channels,mid_channels=mid_channels,num_pred_classes=n_classes).to(device)
            model = ModelCNN(model=modelarch ,n_epochs_stop=30,device=device)#,save_path='saved_model/resNet/'
                    #+dataset_name+"_nFilters_"+str(mid_channels)+"_"+str(i))
            acc = model.train(num_epochs=1000,train_loader=train_dataloader,test_loader=test_dataloader)
            print(i,acc)
            acc_res.append(acc)
        print("\t resnet accuracy is ",np.sum(acc_res)/n_run, acc_res," time was ", (timeit.default_timer() - starttime)/n_run)
        """
        starttime = timeit.default_timer()

        for normal in [True,False]:
            acc_mini=[]
            for i in range(n_run):

                # rocket
                cls = make_pipeline(Rocket(normalise=normal,n_jobs=-1),StandardScaler(),
                                    LogisticRegressionCV(cv = 5, random_state=0, n_jobs = -1,max_iter=1000))
                                    #LogisticRegressionCV(dual=True,solver='liblinear',penalty='l2',max_iter=1000))
                                    #LogisticRegressionCV(solver='newton-cg',multi_class = 'multinomial', class_weight='balanced'))
                                    #RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True))
                cls.fit(train_set,data["y_train"])
                acc = cls.score(test_set,data["y_test"])
                print(normal,i,acc)
                acc_mini.append(acc)
                #dump(cls,"saved_model/rocket/"+dataset_name+"_norm_"+str(normal)+"_"+str(i))
            print("\t rocket normal ",normal," accuracy is ",np.sum(acc_mini)/n_run," time was ", (timeit.default_timer() - starttime)/n_run)

        continue
        starttime = timeit.default_timer()
        for seql in ["fs","clf"]:
            acc_mrSEQL=[]
            for i in range(n_run):
                model = MrSEQLClassifier(seql_mode=seql,symrep=['sax'])
                model.fit(train_set,data["y_train"])
                acc = model.score(test_set,data["y_test"])
                print(acc)
                acc_mrSEQL.append(acc)
                #dump(model,"saved_model/mrSEQL/"+dataset_name+"_seql_"+str(seql)+"_"+str(i))
            print("\t mrSeql ",seql," sax accuracy is ", np.sum(acc_mrSEQL)/n_run," time was ",  (timeit.default_timer() - starttime)/n_run)




if __name__ == "__main__" :
    main()