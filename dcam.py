from load_data import *
from utilities import transform_data4ResNet
import numpy as np
import matplotlib.pyplot as plt
import torch
import timeit
from sklearn.metrics import accuracy_score
import pandas as pd

# importing srcs for dResNet and dCAM
import sys
base_path="./dCAM/src/"
sys.path.insert(0, base_path+'explanation')
sys.path.insert(0, base_path+'models')


#from dcam import *
from CNN_models import *
from DCAM import DCAM

# function took by dCAM notebook to plot the explanation
def plot(dcam,instance,dataset_name,j,true_label,dimension_names):
    plt.figure(figsize=(20,5), dpi=80)
    plt.title('multivariate data series')
    nb_dim = len(instance)
    for i in range(nb_dim):
        plt.subplot(nb_dim,1,1+i)
        plt.plot(instance[i])
        plt.xlim(0,len(instance[i]))
        plt.yticks([0],[dimension_names[i]])

    plt.figure(figsize=(20,5))
    #plt.title('dCAM')
    plt.imshow(dcam,aspect='auto',interpolation=None)
    plt.yticks(list(range(nb_dim)), [el for el in dimension_names])
    file_name = dataset_name+"_"+str(j)+"_GTlabel.png" if true_label else dataset_name+"_"+str(j)+"_OUTlabel.png"
    plt.savefig("explanations/dCAM_results/plots/"+dataset_name+"/"+file_name)
    #plt.colorbar(img)

def main():
    concat = False
    all_data = load_data("MP",concat=concat)

    for dataset_name in all_data.keys():

        # transform data into pytorch format
        train_dataloader,test_dataloader, n_channels, n_classes, device,enc = \
            transform_data4ResNet(all_data[dataset_name],dataset_name, concat=concat)

        # load previously trained dResNet and predict each test set instance
        print("saved_model/resNet/"+dataset_name+"_concat_"+str(concat) )
        modelarch = torch.load("saved_model/resNet/"+dataset_name+"_concat_"+str(concat) )
        resnet = ModelCNN(model=modelarch ,n_epochs_stop=30,device=device)#,save_path='saved_model/resNet/'#+dataset_name+"_nFilters_"+str(mid_channels)+"_"+str(i))
        cnn_output = resnet.predict( test_dataloader )

        # convert back to symbolic representation and get accuracy
        symbolic_output = enc.inverse_transform(cnn_output)
        print(dataset_name,"concat",concat,"accuracy is",accuracy_score(symbolic_output,all_data[dataset_name]["y_test"]))

        # variables used for dCAM
        last_conv_layer = resnet.model._modules['layers'][2]
        fc_layer_name = resnet.model._modules['final']
        testSet_length =  all_data[dataset_name]["X_test"].shape[0]
        target_idxs = enc.transform(all_data[dataset_name]["y_test"])
        explanation= [ {} for i in range (testSet_length)]
        X_test = all_data[dataset_name]["X_test"]
        column_names = X_test.columns.values if type(all_data[dataset_name]['X_train'])==pd.DataFrame else [i for i in range(n_channels)]

        # initialize dCAM object and explain
        dcam = DCAM(resnet.model,device,last_conv_layer=last_conv_layer,fc_layer_name=fc_layer_name)
        starttime = timeit.default_timer()
        for i in range(testSet_length):

            print("explaining ",i,"-th sample of",dataset_name,"out of",testSet_length)

            instance = X_test[i] if type(X_test)==np.ndarray else X_test.values[i]
            if concat:
                instance = np.expand_dims(instance,0)
            gt_label = target_idxs[i]
            output_label = cnn_output[i]

            # CMJ has just 3 channels -> #possibple permutations=6
            nb_permutation = 6 if dataset_name=="CMJ" else 200
            generate_all = True if dataset_name=="CMJ" else False

            try:
                # try to explain the predictions for the ground truth label
                dcam_tl,permutation_success_tl = dcam.run(
                    instance=instance, nb_permutation=nb_permutation, label_instance=gt_label,generate_all=generate_all)
                explanation[i]["dcam_tl"] = dcam_tl
                explanation[i]["permutation_success_tl"] = permutation_success_tl
                plot(dcam_tl,instance,dataset_name,i,True,column_names)
            except IndexError:
                explanation[i]["dcam_tl"] = np.array(-1)
                explanation[i]["permutation_success_tl"] = 0
                sys.stderr.write("index error in ground truth""""""""\n\n")

            try:
                # try to explain the predictions for the ground output label
                dcam_ol,permutation_success_ol = dcam.run(
                    instance=instance, nb_permutation=nb_permutation, label_instance=output_label,generate_all=generate_all)
                explanation[i]["dcam_ol"] = dcam_ol
                explanation[i]["permutation_success_ol"] = permutation_success_ol
                plot(dcam_ol,instance,dataset_name,i,False,column_names)
            except IndexError:
                explanation[i]["dcam_ol"] =  np.array(-1)
                explanation[i]["permutation_success_ol"] = 0
                sys.stderr.write("index error in predictions""""""""\n\n")

            # put in the returned data structure both ground truth and output label previously computed
            explanation[i]["ground_truth_label"] = all_data[dataset_name]["y_test"][i]
            explanation[i]["output_label"] = symbolic_output[i]

        print("average time spent was", (timeit.default_timer() - starttime))
        np.save("explanations/dCAM_results/"+dataset_name+"_explenations",explanation)

if __name__ == "__main__" :
    main()