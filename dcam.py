from load_data import *
from joblib import dump, load
import torch
from utilities import transform_data4ResNet
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from random import randint
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import DataLoader
from utilities import convert2tensor
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import torch
from utilities import interpolation, gen_cube
from torchsummary import summary



base_path="./dCAM/src/"
sys.path.insert(0, base_path+'explanation')
sys.path.insert(0, base_path+'models')


#from dcam import *
from CNN_models import *
from DCAM import DCAM


def main():
    all_data = load_data("synth")

    for dataset_name in all_data.keys():
        train_dataloader,test_dataloader, n_channels, n_classes, device = transform_data4ResNet(all_data[dataset_name],dataset_name)
        modelarch = torch.load("saved_model/resNet/"+dataset_name+"_1.ph")
        resnet = ModelCNN(model=modelarch ,n_epochs_stop=30,device=device)#,save_path='saved_model/resNet/'
        #+dataset_name+"_nFilters_"+str(mid_channels)+"_"+str(i))
        #output = resnet.predict( test_dataloader )

        instance = all_data[dataset_name]["X_train"][0]
        label_instance = 1

        instance_to_try = Variable(
            torch.tensor(
                instance.reshape(
                    (1,1,n_channels,100))).float().to(device),
            requires_grad=True)


        # dResNet
        last_conv_layer = resnet.model._modules['layers'][2]
        fc_layer_name = resnet.model._modules['final']


        print(instance_to_try.shape)
        DCAM_m = DCAM(resnet.model,device,last_conv_layer=last_conv_layer,fc_layer_name=fc_layer_name)
        dcam,permutation_success = DCAM_m.run(
            instance=instance,
            nb_permutation=200,
            label_instance=label_instance)
        print(dcam.shape,dcam)


if __name__ == "__main__" :
    main()