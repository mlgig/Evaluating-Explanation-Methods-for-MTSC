from sktime.utils.data_io  import load_from_arff_to_dataframe, load_from_tsfile_to_dataframe
from utilities import interpolation
import numpy as np
import os

def load_data(data_name):
    data = {}
    if data_name=="synth":
        base_path = "./data/synth_data/data/"
        middle= "_F_20_TS_100_Positional_"
        for generation_kinds in ["PseudoPeriodic","GaussianProcess","AutoRegressive"]:
            for Positional in ["False"]:
                train_meta = np.load(os.path.join( base_path,"SimulatedTrainingMetaData_RareTime_"+generation_kinds+middle+Positional+".npy"),allow_pickle=True).item()
                test_meta =  np.load(os.path.join( base_path,"SimulatedTestingMetaData_RareTime_"+generation_kinds+middle+Positional+".npy"),allow_pickle=True).item()
                data[generation_kinds+"_Positional_"+Positional] = {
                    "X_train" : np.transpose(np.load(os.path.join(
                        base_path,"SimulatedTrainingData_RareTime_"+generation_kinds+middle+Positional+".npy")) ,(0,2,1)),
                    "X_test" :  np.transpose(np.load(os.path.join(
                         base_path,"SimulatedTestingData_RareTime_"+generation_kinds+middle+Positional+".npy")),(0,2,1)),
                    "y_train" : train_meta['Targets'],
                    "y_test" : test_meta['Targets'],
                    "meta_train" : train_meta,
                    "meta_test" : test_meta,
                }
                del data[generation_kinds+"_Positional_"+Positional]["meta_train"]['Targets']
                del data[generation_kinds+"_Positional_"+Positional]["meta_test"]['Targets']
    elif data_name=="CMJ":
        base_path="data/CounterMovementJump/"
        name = "CounterMovementJump"
        CMJ = {}
        CMJ["X_train"], CMJ["y_train"] = load_from_arff_to_dataframe(os.path.join(base_path,name+"_TRAIN.arff"),replace_missing_vals_with='0')
        CMJ["X_test"], CMJ["y_test"] = load_from_arff_to_dataframe(os.path.join(base_path,name+"_TEST.arff"),replace_missing_vals_with='0')
        data["CMJ"] = CMJ
    elif data_name=="MP":
        base_path="/home/davide/Downloads/PoseEstimation-20221206T155730Z-001/PoseEstimation/OpenPosev1.7/MP/Unnormalized8VariableLength"

        MP = {}
        X_train, y_train = load_from_tsfile_to_dataframe(os.path.join(base_path,"TRAIN_default_X.ts"))
        X_test, y_test = load_from_tsfile_to_dataframe(os.path.join(base_path,"TEST_default_X.ts"))

        MP["X_train"] = interpolation(X_train.values.tolist(),max_length=161,n_var=8)
        MP["X_test"] = interpolation(X_test.values.tolist(),max_length=161,n_var=8)
        MP["y_train"] = y_train
        MP["y_test"] = y_test

        data["MP"] = MP
    else:
        raise ValueError("only synth,CMJ or MP are valid dataset values")

    return data
