from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import LogisticRegressionCV
from  sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from load_data import load_data
import sys
sys.path.insert(0, 'timeXplain')
import timexplain as tx
import numpy as np
import timeit

def main():
    # load dataset and select just one channel
    all_data = load_data("MP")
    for dataset_name in all_data.keys():
        for channel in (9,10):
            # for last informative channel and for fist UNinformative channel
            X_train = np.squeeze(all_data[dataset_name]["X_train"][:,channel:channel+1,:],axis=1)
            X_test =  np.squeeze( all_data[dataset_name]["X_test"][:,channel:channel+1,:],axis=1)
            y_train = all_data[dataset_name]["y_train"]
            y_test =   all_data[dataset_name]["y_test"]
            size_x = X_train.shape[-1]
            # train classifier
            cls = make_pipeline(Rocket(normalise=False,n_jobs=-1),StandardScaler(),
                        LogisticRegressionCV(cv = 5, random_state=0, n_jobs = -1,max_iter=1000))
            cls.fit( X_train,y_train )
            acc = cls.score( X_test,y_test)
            print("accuracy on",dataset_name,"was",acc)

            # explain
            starttime = timeit.default_timer()
            rocket_om = tx.om.TimeSliceOmitter(size_x, time_slicing=10, x_repl=tx.om.x_sample)
            shap_explainer = tx.om.KernelShapExplainer(rocket_om, cls.predict_proba, X_bg=X_test, y_bg=y_test, n_samples=500, n_builds=5, bgcs=True)
            exps = []
            for i in range(X_test.shape[-1]):
                X_specimen = X_test[i]
                y_specimen = y_test[i]
                rocket_expl = shap_explainer.explain(X_specimen)
                exps.append(rocket_expl.impacts)
            print("time was", timeit.default_timer() - starttime)
            file_name = "rocket_shap_results/"+dataset_name+"_channel_"+str(channel)+".npy"
            exps = np.array(exps)
            np.save(file_name,exps)

        break
if __name__ == "__main__" :
    main()