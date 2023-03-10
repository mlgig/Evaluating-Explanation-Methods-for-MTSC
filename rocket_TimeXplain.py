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
from sklearn.metrics import accuracy_score
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from sktime.transformations.series.adapt import _from_series_to_2d_numpy  ###still a nested series
from sktime.transformations.panel.compose import ColumnConcatenator

def trainAndExplain(X_train,y_train,X_test,y_test,channel_name,size_x,n_slices,dataset_name,curr_section):
    # train classifier, save the predictions and get accuracy
    starttime = timeit.default_timer()
    cls = make_pipeline(Rocket(normalise=False,n_jobs=-1),StandardScaler(),
                        LogisticRegressionCV(cv = 5, random_state=0, n_jobs = -1,max_iter=1000))
    cls.fit( X_train,y_train )
    outputs = cls.predict( X_test)
    acc = accuracy_score(outputs,y_test)
    print(dataset_name,": time to train was ",timeit.default_timer() - starttime,
          "accuracy on channel",channel_name,"was",acc)

    # explain
    starttime = timeit.default_timer()
    rocket_om = tx.om.TimeSliceOmitter(size_x, time_slicing=n_slices, x_repl=tx.om.x_sample)
    shap_explainer = tx.om.KernelShapExplainer(rocket_om, cls.predict_proba, X_bg=X_test, y_bg=y_test, n_samples=500, n_builds=5, bgcs=True)

    for i in range(X_test.shape[0]):
        # save in the dictionary model predictions, ground truth labels and explanation arrays
        X_specimen = X_test[i]

        curr_section["model_outputs"][i] = outputs[i]
        curr_section["ground_truth_labels"][i]  = y_test[i]
        rocket_expl = shap_explainer.explain(X_specimen)
        curr_section["exps"][i]  = rocket_expl.impacts
        print("explained item",i)
    print("Time to explain",channel_name,"was", timeit.default_timer() - starttime)

    return curr_section

def main():
    # load dataset and select just one channel
    all_data = load_data("synth")
    for dataset_name in all_data.keys():
        if dataset_name=='PseudoPeriodic_Positional_False':
            continue

        # load datasets (need to convert into numpy arrays to use timeXplain)
        X_train_orig =  all_data[dataset_name]['X_train'] if type(all_data[dataset_name]['X_train'])==np.ndarray else \
             from_nested_to_3d_numpy( all_data[dataset_name]['X_train'])
        X_test_orig =  all_data[dataset_name]['X_test'] if type(all_data[dataset_name]['X_test'])==np.ndarray else \
            from_nested_to_3d_numpy( all_data[dataset_name]['X_test'])
        y_train = all_data[dataset_name]["y_train"]
        y_test = all_data[dataset_name]["y_test"]

        # get infos
        n_channels = X_train_orig.shape[1]
        # in case they are available (i.e. original dataset is  pandas dataframe) get the column
        channel_names = ["dim "+str(i) for i in range(n_channels)] if type(all_data[dataset_name]['X_test'])==np.ndarray else \
            all_data[dataset_name]['X_test'].columns.values
        test_length = X_test_orig.shape[0]
        n_classes = len(np.unique(y_test))

        # set parameters and data structure for results
        n_slices = 20
        column_concat = False
        file_path = "explanations/rocket_shap_results/"

        # data structure containing all infos
        if column_concat:
            exps = {
                "model_outputs" : np.zeros(shape=(test_length)),
                "ground_truth_labels" :np.zeros(shape=(test_length)),
                "exps" : np.zeros(shape=(test_length,n_classes,n_slices))
            }
        else:
            exps = {
                "model_outputs" : np.zeros(shape=(n_channels,test_length)),
                "ground_truth_labels" :np.empty(shape=(n_channels,test_length)),
                "exps" : np.empty(shape=(test_length,n_channels,n_classes,n_slices))
            }

        # reshape/take a data slice and predict
        if column_concat:
            #
            cc = ColumnConcatenator()
            X_train = np.squeeze( from_nested_to_3d_numpy( cc.fit_transform(X_train_orig)), axis=1)
            X_test =   np.squeeze( from_nested_to_3d_numpy( cc.transform(X_test_orig)), axis=1)
            size_x = X_train_orig.shape[-1] *  X_train_orig.shape[-2]

            exps = trainAndExplain(X_train,y_train,X_test,y_test,channel_name="concatenated",
                            size_x=size_x,n_slices=n_slices,curr_section=exps,dataset_name=dataset_name)
            file_name = dataset_name+"_Concatenated_20"+str(n_slices)+".npy"
            np.save(file_path+file_name,exps)
        else:
            for channel in range(n_channels):
                X_train = np.squeeze( X_train_orig[:, channel:(channel+1), :])
                X_test=  np.squeeze( X_test_orig[:, channel:(channel+1), :] )
                size_x = X_train_orig.shape[-1]
                ch_name = channel_names[channel]
                # in this case we select slices of explanation dictionary
                current_slice = {
                    "model_outputs" : exps["model_outputs"][channel,:],
                    "ground_truth_labels" : exps["ground_truth_labels"][channel,:],
                    "exps" : exps["exps"][:,channel,:,:]
                }
                # explain
                current_slice=trainAndExplain(X_train,y_train,X_test,y_test,channel_name=ch_name,
                                size_x=size_x,n_slices=n_slices,curr_section=current_slice,dataset_name=dataset_name)

                # and put results into the original positions
                exps["model_outputs"][channel,:] = current_slice["model_outputs"]
                exps["ground_truth_labels"][channel,:] = current_slice["ground_truth_labels"]
                exps["exps"][:,channel,:,:] = current_slice["exps"]

            file_name = dataset_name+"_channelByChannel.npy"
            np.save(file_path+file_name,exps)

        break
if __name__ == "__main__" :
    main()