from sktime.utils.data_io  import load_from_arff_to_dataframe, load_from_tsfile_to_dataframe
from utilities import interpolation
import numpy as np
import os

def load_data(data_name):
    data = {}
    if data_name=="synth":
        base_path = "./data/synth_data/data/"
        middle= "_F_20_TS_100_Moving_"
        for generation_kinds in ["AutoRegressive","PseudoPeriodic","GaussianProcess"]:
            for moving in ["True"]:
                train_meta = np.load(os.path.join( base_path,"SimulatedTrainingMetaData_RareTime_"+generation_kinds+middle+moving+".npy"),allow_pickle=True).item()
                test_meta =  np.load(os.path.join( base_path,"SimulatedTestingMetaData_RareTime_"+generation_kinds+middle+moving+".npy"),allow_pickle=True).item()
                data[generation_kinds+"_moving_"+moving] = {
                    "X_train" : np.load(os.path.join( base_path,"SimulatedTrainingData_RareTime_"+generation_kinds+middle+moving+".npy")),
                    "X_test" : np.load(os.path.join( base_path,"SimulatedTestingData_RareTime_"+generation_kinds+middle+moving+".npy")),
                    "y_train" : train_meta['Targets'],
                    "y_test" : test_meta['Targets'],
                    "meta_train" : train_meta,
                    "meta_test" : test_meta,
                }
                del data[generation_kinds+"_moving_"+moving]["meta_train"]['Targets']
                del data[generation_kinds+"_moving_"+moving]["meta_test"]['Targets']
    elif data_name=="CMJ":
        # TODO replace nan with 0 values
        base_path="data/CounterMovementJump/"
        name = "CounterMovementJump"
        CMJ = {}
        CMJ["X_train"], CMJ["y_train"] = load_from_arff_to_dataframe(os.path.join(base_path,name+"_TRAIN.arff"),replace_missing_vals_with='0')
        CMJ["X_test"], CMJ["y_test"] = load_from_arff_to_dataframe(os.path.join(base_path,name+"_TEST.arff"),replace_missing_vals_with='0')
        data["CMJ"] = CMJ
    elif data_name=="MP":
        # TODO V1.7 fULLuNNORMALIZED25xy and try the 8 variables
        # From the 25 body parts dataset, you can load it and only select the 8 body parts Ashish uses in the paper,
        # ie wrists, elbows, shoulders and hips. Then just work with this smaller dataframe

        #base_path="/home/davide/Downloads/PoseEstimation-20221206T155730Z-001/PoseEstimation/OpenPosev1.4/MP/FullUnnormalized14-OpenPosev14"
        base_path="/home/davide/Downloads/PoseEstimation-20221206T155730Z-001/PoseEstimation/OpenPosev1.4/MP/FullUnnormalized25/"
        base_path="/home/davide/Downloads/PoseEstimation-20221206T155730Z-001/PoseEstimation/OpenPosev1.7/MP/Unnormalized8VariableLength"

        MP = {}
        #MP["X_train"], MP["y_train"] = load_from_tsfile_to_dataframe(os.path.join(base_path,"TRAIN_X.ts"))
        #MP["X_test"], MP["y_test"] = load_from_tsfile_to_dataframe(os.path.join(base_path,"TEST_X.ts"))


        X_train, y_train = load_from_tsfile_to_dataframe(os.path.join(base_path,"TRAIN_default_X.ts"))
        X_test, y_test = load_from_tsfile_to_dataframe(os.path.join(base_path,"TEST_default_X.ts"))
        MP["X_train"] = interpolation(X_train.values.tolist(),161,8)
        MP["X_test"] = interpolation(X_test.values.tolist(),161,8)
        MP["y_train"] = y_train
        MP["y_test"] = y_test

        data["MP"] = MP
    else:
        raise ValueError("only synth,CMJ or MP are valid dataset values")

    return data
