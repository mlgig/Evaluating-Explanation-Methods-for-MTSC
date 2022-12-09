from sktime.utils.data_io  import load_from_arff_to_dataframe
import numpy as np
import os

def load_data(data_name):
    data = {}
    if data_name=="synth":
        base_path = "/home/davide/Desktop/synth_data/data/"
        #TODO not a constant here!
        n_f = "20"
        print(n_f)
        middle= "_F_"+n_f+"_TS_100_Positional_"
        for generation_kinds in ["AutoRegressive","PseudoPeriodic","GaussianProcess"]:
            #TODO reinsert "True" in the line below
            for positional in ["False"]:
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
        pass
    else:
        raise ValueError("only synth,CMJ or MP are valid dataset values")

    return data
