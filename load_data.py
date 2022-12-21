from sktime.utils.data_io  import load_from_arff_to_dataframe, load_from_tsfile_to_dataframe
import numpy as np
import os

def load_data(data_name):
    data = {}
    if data_name=="synth":
        base_path = "./data/synth_data/data/"
        middle= "_F_20_TS_100_Positional_"
        for generation_kinds in ["AutoRegressive","PseudoPeriodic","GaussianProcess"]:
            for positional in ["False","True"]:
                train_meta = np.load(os.path.join( base_path,"SimulatedTrainingMetaData_RareTime_"+generation_kinds+middle+positional+".npy"),allow_pickle=True).item()
                test_meta =  np.load(os.path.join( base_path,"SimulatedTestingMetaData_RareTime_"+generation_kinds+middle+positional+".npy"),allow_pickle=True).item()
                data[generation_kinds+"_positional_"+positional] = {
                    "X_train" : np.load(os.path.join( base_path,"SimulatedTrainingData_RareTime_"+generation_kinds+middle+positional+".npy")),
                    "X_test" : np.load(os.path.join( base_path,"SimulatedTestingData_RareTime_"+generation_kinds+middle+positional+".npy")),
                    "y_train" : train_meta['Targets'],
                    "y_test" : test_meta['Targets'],
                    "meta_train" : train_meta,
                    "meta_test" : test_meta,
                }
                del data[generation_kinds+"_positional_"+positional]["meta_train"]['Targets']
                del data[generation_kinds+"_positional_"+positional]["meta_test"]['Targets']
    elif data_name=="CMJ":
        base_path="data/CounterMovementJump/"
        name = "CounterMovementJump"
        CMJ = {}
        CMJ["X_train"], CMJ["y_train"] = load_from_arff_to_dataframe(os.path.join(base_path,name+"_TEST.arff"))
        CMJ["X_test"], CMJ["y_test"] = load_from_arff_to_dataframe(os.path.join(base_path,name+"_TEST.arff"))
        data["CMJ"] = CMJ
    elif data_name=="MP":
        #base_path="/home/davide/Downloads/PoseEstimation-20221206T155730Z-001/PoseEstimation/OpenPosev1.4/MP/FullUnnormalized14-OpenPosev14"
        base_path="/home/davide/Downloads/PoseEstimation-20221206T155730Z-001/PoseEstimation/OpenPosev1.7/MP/FullUnnormalized25/"
        MP = {}
        #MP["X_train"], MP["y_train"] = load_from_tsfile_to_dataframe(os.path.join(base_path,"TRAIN_X.ts"))
        #MP["X_test"], MP["y_test"] = load_from_tsfile_to_dataframe(os.path.join(base_path,"TEST_X.ts"))
        MP["X_train"], MP["y_train"] = load_from_tsfile_to_dataframe(os.path.join(base_path,"TRAIN_default_X.ts"))
        MP["X_test"], MP["y_test"] = load_from_tsfile_to_dataframe(os.path.join(base_path,"TEST_default_X.ts"))
        data["MP"] = MP
    else:
        raise ValueError("only synth,CMJ or MP are valid dataset values")

    return data
