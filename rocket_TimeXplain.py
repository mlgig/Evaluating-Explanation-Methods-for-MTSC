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
import pandas as pd
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from sktime.transformations.series.adapt import _from_series_to_2d_numpy  ###still a nested series
from sktime.transformations.panel.compose import ColumnConcatenator

def main():
    # load dataset and select just one channel
    all_data = load_data("synth")
    for dataset_name in all_data.keys():

        # load dataset
        X_train_orig =  all_data[dataset_name]['X_train'] if type(all_data[dataset_name]['X_train'])==np.ndarray else \
             from_nested_to_3d_numpy( all_data[dataset_name]['X_train'])
        X_test_orig =  all_data[dataset_name]['X_test'] if type(all_data[dataset_name]['X_test'])==np.ndarray else \
            from_nested_to_3d_numpy( all_data[dataset_name]['X_test'])
        y_train = all_data[dataset_name]["y_train"]
        y_test =   all_data[dataset_name]["y_test"]

        # get infos
        n_channels = X_train_orig.shape[1]
        channel_names = ["dim "+str(i) for i in range(n_channels)] if type(all_data[dataset_name]['X_test'])==np.ndarray else \
            all_data[dataset_name]['X_test'].columns.values
        test_length = X_test_orig.shape[0]
        timeSeries_Length =  X_test_orig.shape[-1]
        n_classes = len(np.unique(y_test))

        n_slices = 10
        # data structure containing all infos
        exps = {
            #"concatenated" : np.zeros(shape=(test_length,n_classes,n_channels)),
            "channelByChannel" : np.zeros(shape=(test_length,n_channels,n_classes,n_slices))
        }

        for channel in range(-1,n_channels,1):
            starttime = timeit.default_timer()
            if channel==-1:
                continue
                # if channel == -1 concatenate all the channels together
                cc = ColumnConcatenator()
                X_train = np.squeeze( from_nested_to_3d_numpy( cc.fit_transform(X_train_orig)), axis=1)
                X_test =   np.squeeze( from_nested_to_3d_numpy( cc.transform(X_test_orig)), axis=1)
                size_x = X_train_orig.shape[-1] *  X_train_orig.shape[-2]
            else:
                X_train = np.squeeze( X_train_orig[:, channel:(channel+1), :])
                X_test=  np.squeeze( X_test_orig[:, channel:(channel+1), :] )
                size_x = X_train_orig.shape[-1]

            # train classifier and get accuracy
            cls = make_pipeline(Rocket(normalise=False,n_jobs=-1),StandardScaler(),
                        LogisticRegressionCV(cv = 5, random_state=0, n_jobs = -1,max_iter=1000))
            cls.fit( X_train,y_train )
            acc = cls.score( X_test,y_test)
            channel_name = "all" if channel==-1 else channel_names[channel]
            print(dataset_name,": time to train was ",timeit.default_timer() - starttime,
                  "accuracy on channel",channel,channel_name,"was",acc)

            # explain
            starttime = timeit.default_timer()
            rocket_om = tx.om.TimeSliceOmitter(size_x, time_slicing=n_slices, x_repl=tx.om.x_sample)
            shap_explainer = tx.om.KernelShapExplainer(rocket_om, cls.predict_proba, X_bg=X_test, y_bg=y_test, n_samples=500, n_builds=5, bgcs=True)

            for i in range(test_length):
                X_specimen = X_test[i]
                y_specimen = y_test[i]
                rocket_expl = shap_explainer.explain(X_specimen)
                print("explained item",i)
                if channel==-1:
                    exps["concatenated"][i]=rocket_expl.impacts
                else:
                    exps["channelByChannel"][i][channel]=rocket_expl.impacts
            print("Time to explain item number",str(i),"was", timeit.default_timer() - starttime)

        file_name = "explanations/rocket_shap_results/"+dataset_name+".npy"
        np.save(file_name,exps)

        break
if __name__ == "__main__" :
    main()