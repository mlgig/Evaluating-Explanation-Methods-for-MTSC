from sklearn.pipeline import make_pipeline
from load_data import load_data
import sys
sys.path.insert(0, 'timeXplain')
import timexplain as tx
import numpy as np
import timeit
from sklearn.metrics import accuracy_score
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from sklearn.preprocessing import FunctionTransformer
from joblib import  load
import sys
sys.path.insert(0, 'timeXplain')


def trainAndExplain(X_train,y_train,X_test,y_test,channel_name,size_x,n_slices,dataset_name,curr_section):

    # try to load previously trained model and test accuracy
    saved_steps = load("saved_model/rocket/"+dataset_name+"_"+channel_name)
    cls = make_pipeline(  FunctionTransformer( lambda x: np.expand_dims(x,1), validate=True),
                          saved_steps[0],saved_steps[1],saved_steps[2])

    outputs = cls.predict( X_test)
    acc = accuracy_score(outputs,y_test)
    print(dataset_name,"accuracy on channel",channel_name,"was",acc, X_test.shape)

    # initialize explainers and explain
    rocket_om = tx.om.TimeSliceOmitter(size_x, time_slicing=n_slices, x_repl=tx.om.x_sample)
    shap_explainer = tx.om.KernelShapExplainer(rocket_om, cls.predict_proba, X_bg=X_test, y_bg=y_test, n_samples=500, n_builds=5, bgcs=True)

    for i in range(X_test.shape[0]):
        # save in the dictionary model predictions, ground truth labels and explanation arrays
        starttime = timeit.default_timer()
        curr_idx = i
        print("explaining",curr_idx,channel_name)
        X_specimen = X_test[curr_idx]

        # save both ground truth and predicted labels
        curr_section["model_outputs"][curr_idx] = outputs[i]
        curr_section["ground_truth_labels"][curr_idx]  = y_test[i]
        # explain and save the result
        rocket_expl = shap_explainer.explain(X_specimen)
        curr_section["exps"][curr_idx]  = rocket_expl.impacts
        print("explained item",curr_idx, timeit.default_timer() - starttime)

    return curr_section

def main():

    column_concat = True
    all_data = load_data("synth",concat=column_concat)
    for dataset_name in all_data.keys():

        # load datasets (need to convert into numpy arrays to use timeXplain)
        X_train =  all_data[dataset_name]['X_train'] if type(all_data[dataset_name]['X_train'])==np.ndarray else \
            from_nested_to_3d_numpy( all_data[dataset_name]['X_train'])
        X_test =  all_data[dataset_name]['X_test'] if type(all_data[dataset_name]['X_test'])==np.ndarray else \
            from_nested_to_3d_numpy( all_data[dataset_name]['X_test'])
        y_train = all_data[dataset_name]["y_train"]
        y_test = all_data[dataset_name]["y_test"]

        # get infos
        n_channels = X_train.shape[1]
        test_length = X_test.shape[0]
        n_classes = len(np.unique(y_test))
        # in case they are available (i.e. original dataset is  pandas dataframe) get column names
        channel_names = ["dim_"+str(i) for i in range(n_channels)] if type(all_data[dataset_name]['X_test'])==np.ndarray else \
            all_data[dataset_name]['X_test'].columns.values

        # set parameters and data structure for results
        n_slices = 20
        file_path = "explanations/rocket_shap_results/"
        data_type = 'int' if dataset_name=="synth" else '<U3'

        # data structure containing all infos
        if column_concat:
            # initialise datastructure for explanations
            exps = {
                "model_outputs" : np.zeros(shape=(test_length),dtype=data_type),
                "ground_truth_labels" :np.zeros(shape=(test_length),dtype=data_type),
                "exps" : np.zeros(shape=(test_length,n_classes,n_slices))
            }

            # concatenate
            size_x = X_train.shape[1]
            exps = trainAndExplain(X_train,y_train,X_test,y_test,channel_name="concatenated",
                                   size_x=size_x,n_slices=n_slices,curr_section=exps,dataset_name=dataset_name)
            file_name = dataset_name+"_Concatenated_"+str(n_slices)+".npy"
        else:
            # initialise datastructure for explanations
            exps = {
                "model_outputs" : np.zeros(shape=(n_channels,test_length),dtype=data_type),
                "ground_truth_labels" :np.empty(shape=(n_channels,test_length),dtype=data_type),
                "exps" : np.empty(shape=(test_length,n_channels,n_classes,n_slices))
            }

            # and explain channel by channel
            for channel in range(0,n_channels):
                print("channel",channel)
                # prepare single channel datasets
                X_train =  np.squeeze( X_train[:, channel:(channel+1), :] )
                X_test=  np.squeeze( X_test[:, channel:(channel+1), :] )
                size_x = X_train.shape[-1]
                ch_name = channel_names[channel]

                # prepare a temporary data structure
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
                file_name = dataset_name+"_ChannelByChannel_"+str(n_slices)+".npy"

        np.save(file_path+file_name,exps)

if __name__ == "__main__" :
    main()