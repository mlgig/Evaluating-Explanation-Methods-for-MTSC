from sktime.utils.data_io  import load_from_arff_to_dataframe
import numpy as np
import os

def load_data(data_name):
    data = {}
    if data_name=="synth":
        base_path = "data/synth_data/data/"
        middle= "_F_10_TS_100_Positional_"
        for generation_kinds in ["AutoRegressive","GaussianProcess","PseudoPeriodic"]:
            #TODO create github repo
            for positional in ["True","False"]:
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
        data["X_train"], data["y_train"] = load_from_arff_to_dataframe(os.path.join(base_path,name+"_TEST.arff"))
        data["X_test"], data["y_test"] = load_from_arff_to_dataframe(os.path.join(base_path,name+"_TEST.arff"))
    elif data_name=="MP":
        pass
    else:
        raise ValueError("only synth,CMJ or MP are valid dataset values")

    return data
