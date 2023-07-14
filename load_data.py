from sktime.datasets  import  load_from_tsfile_to_dataframe
import numpy as np
import os
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
import pandas as pd

def load_data(data_name,concat):
    data = {}
    if data_name=="synth":
        base_path = "./data/synth_data/data/"

        middle= "_F_20_TS_100_Positional_"
        for generation_kinds in ["PseudoPeriodic","GaussianProcess","AutoRegressive"]:
            for Positional in ["False"]:
                train_meta = np.load(os.path.join( base_path,"SimulatedTrainingMetaData_RareTime_"+generation_kinds+middle+Positional+".npy"),allow_pickle=True).item()
                test_meta =  np.load(os.path.join( base_path,"SimulatedTestingMetaData_RareTime_"+generation_kinds+middle+Positional+".npy"),allow_pickle=True).item()
                k = generation_kinds+"_Positional_"+Positional
                data[k] = {
                    "X_train" : np.transpose(np.load(os.path.join(
                        base_path,"SimulatedTrainingData_RareTime_"+generation_kinds+middle+Positional+".npy")) ,(0,2,1)),
                    "X_test" :  np.transpose(np.load(os.path.join(
                        base_path,"SimulatedTestingData_RareTime_"+generation_kinds+middle+Positional+".npy")),(0,2,1)),
                    "y_train" : train_meta['Targets'],
                    "y_test" : test_meta['Targets'],
                    "meta_train" : train_meta,
                    "meta_test" : test_meta,
                }
                del data[k]["meta_train"]['Targets']
                del data[k]["meta_test"]['Targets']
                if concat:
                    data[k]["X_train" ] = data[k]["X_train" ].reshape(data[k]["X_train" ].shape[0],-1)
                    data[k]["X_test" ] = data[k]["X_test" ].reshape(data[k]["X_test" ].shape[0],-1)

    elif data_name=="CMJ":
        base_path="data/CounterMovementJump/"
        name = "resampled.npy"
        CMJ = np.load(base_path+name,allow_pickle=True).item()
        data["CMJ"] = CMJ
        if concat:
            CMJ["X_train"] = CMJ["X_train"].reshape(CMJ["X_train"].shape[0],-1)
            CMJ["X_test"] = CMJ["X_test"].reshape(CMJ["X_test"].shape[0],-1)


        #CMJ = np.load("/home/davide/workspace/PhD/Trang/first_experiment/CMJ_resampled_390.npy",allow_pickle=True).item()
        #if concat:
        #    CMJ["X_train"] = CMJ["X_train"].reshape(CMJ["X_train"].shape[0],-1)
        #    CMJ["X_test"] =  CMJ["X_test"].reshape(CMJ["X_test"].shape[0],-1)
        #data["CMJ"] = CMJ
    elif data_name=="MP":
        base_path="data/Military press/Full25BodyParts/Full25BodyParts/"
        MP = {}
        X_train, y_train = load_from_tsfile_to_dataframe(os.path.join(base_path,"TRAIN_full_X.ts"))#"TRAIN_default_X.ts"))
        X_test, y_test = load_from_tsfile_to_dataframe(os.path.join(base_path,"TEST_full_X.ts"))#"TEST_default_X.ts"))
        X_train.columns = ['Nose_X', 'Neck_X', 'RShoulder_X', 'RElbow_X', 'RWrist_X', 'LShoulder_X', 'LElbow_X', 'LWrist_X', 'MidHip_X', 'RHip_X', 'RKnee_X', 'RAnkle_X', 'LHip_X', 'LKnee_X', 'LAnkle_X', 'REye_X', 'LEye_X', 'REar_X', 'LEar_X', 'LBigToe_X', 'LSmallToe_X', 'LHeel_X', 'RBigToe_X', 'RSmallToe_X', 'RHeel_X', 'Nose_Y', 'Neck_Y', 'RShoulder_Y', 'RElbow_Y', 'RWrist_Y', 'LShoulder_Y', 'LElbow_Y', 'LWrist_Y', 'MidHip_Y', 'RHip_Y', 'RKnee_Y', 'RAnkle_Y', 'LHip_Y', 'LKnee_Y', 'LAnkle_Y', 'REye_Y', 'LEye_Y', 'REar_Y', 'LEar_Y', 'LBigToe_Y', 'LSmallToe_Y', 'LHeel_Y', 'RBigToe_Y', 'RSmallToe_Y', 'RHeel_Y']
        X_test.columns = ['Nose_X', 'Neck_X', 'RShoulder_X', 'RElbow_X', 'RWrist_X', 'LShoulder_X', 'LElbow_X', 'LWrist_X', 'MidHip_X', 'RHip_X', 'RKnee_X', 'RAnkle_X', 'LHip_X', 'LKnee_X', 'LAnkle_X', 'REye_X', 'LEye_X', 'REar_X', 'LEar_X', 'LBigToe_X', 'LSmallToe_X', 'LHeel_X', 'RBigToe_X', 'RSmallToe_X', 'RHeel_X', 'Nose_Y', 'Neck_Y', 'RShoulder_Y', 'RElbow_Y', 'RWrist_Y', 'LShoulder_Y', 'LElbow_Y', 'LWrist_Y', 'MidHip_Y', 'RHip_Y', 'RKnee_Y', 'RAnkle_Y', 'LHip_Y', 'LKnee_Y', 'LAnkle_Y', 'REye_Y', 'LEye_Y', 'REar_Y', 'LEar_Y', 'LBigToe_Y', 'LSmallToe_Y', 'LHeel_Y', 'RBigToe_Y', 'RSmallToe_Y', 'RHeel_Y']
        column_names = ['RShoulder_X', 'RElbow_X', 'RWrist_X', 'LShoulder_X', 'LElbow_X', 'LWrist_X', 'MidHip_X', 'RHip_X','LHip_X',
                        'RShoulder_Y', 'RElbow_Y', 'RWrist_Y', 'LShoulder_Y', 'LElbow_Y', 'LWrist_Y', 'MidHip_Y', 'RHip_Y', 'LHip_Y']
        columns_subset = [ 'RShoulder_Y', 'RElbow_Y', 'RWrist_Y', 'LShoulder_Y', 'LElbow_Y', 'LWrist_Y', 'RHip_Y', 'LHip_Y']

        if concat:
            MP["X_train"] = from_nested_to_3d_numpy( X_train[columns_subset] ).reshape(X_train.shape[0],-1)
            MP["X_test"] = from_nested_to_3d_numpy( X_test[columns_subset] ).reshape(X_test.shape[0],-1)
        else:
            MP["X_train"] = X_train[columns_subset]
            MP["X_test"] = X_test[columns_subset]

        MP["y_train"] = y_train
        MP["y_test"] = y_test
        data["MP"] = MP

    else:
        raise ValueError("only synth,CMJ or MP are valid dataset values")



    return data