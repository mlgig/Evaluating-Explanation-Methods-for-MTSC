from load_data import *
from utilities import transform_data4ResNet
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch

#TODO is there a better way to import these files?
base_path="./dCAM/src/"
sys.path.insert(0, base_path+'explanation')
sys.path.insert(0, base_path+'models')


#from dcam import *
from CNN_models import *
from DCAM import DCAM



def plot(dcam,instance,dataset_name,j,true_label):
    plt.figure(figsize=(20,5), dpi=80)
    plt.title('multivariate data series')
    nb_dim = len(instance)
    for i in range(nb_dim):
        plt.subplot(nb_dim,1,1+i)
        plt.plot(instance[i])
        plt.xlim(0,len(instance[i]))
        plt.yticks([0],["Dim {}".format(i)])

    plt.figure(figsize=(20,5))
    #plt.title('dCAM')
    plt.imshow(dcam,aspect='auto',interpolation=None)
    plt.yticks(list(range(nb_dim)), ["Dim {}".format(i) for i in range(nb_dim)])
    file_name = dataset_name+"_"+str(j)+"_GTlabel.png" if true_label else dataset_name+"_"+str(j)+"_OUTlabel.png"
    plt.savefig("dCCAM_results/plots/"+file_name)
    #plt.colorbar(img)

def main():
    all_data = load_data("synth")

    for dataset_name in all_data.keys():
        train_dataloader,test_dataloader, n_channels, n_classes, device,y_test = transform_data4ResNet(all_data[dataset_name],dataset_name)
        modelarch = torch.load("saved_model/resNet/"+dataset_name+".ph")
        resnet = ModelCNN(model=modelarch ,n_epochs_stop=30,device=device)#,save_path='saved_model/resNet/'
        #+dataset_name+"_nFilters_"+str(mid_channels)+"_"+str(i))
        output = resnet.predict( test_dataloader )


        # explain
        last_conv_layer = resnet.model._modules['layers'][2]
        fc_layer_name = resnet.model._modules['final']

        timeseries_length = all_data[dataset_name]["X_test"].shape[-1]
        testSet_length =  all_data[dataset_name]["X_test"].shape[0]
        explanation= [ {} for i in range (testSet_length)]

        for i in range(testSet_length):
            #TODO handle the case in which none of the permutations are according to the labels
            #TODO check this warning
            # MatplotlibDeprecationWarning: Auto-removal of overlapping axes is deprecated since 3.6 and will be removed two minor releases later; explicitly call ax.remove() as needed.
            #   plt.subplot(nb_dim,1,1+i)
            print("explaining ",i,"-th sample of",dataset_name,"out of",testSet_length)

            instance = all_data[dataset_name]["X_test"][i]
            gt_label = all_data[dataset_name]["y_test"][i]
            output_label = output[i]

            dcam = DCAM(resnet.model,device,last_conv_layer=last_conv_layer,fc_layer_name=fc_layer_name)

            try:
                dcam_tl,permutation_success_tl = dcam.run(
                    instance=instance, nb_permutation=200, label_instance=gt_label)
                explanation[i]["dcam_tl"] = dcam_tl
                explanation[i]["permutation_success_tl"] = permutation_success_tl
                plot(dcam_tl,instance,dataset_name,i,True)
            except IndexError:
                explanation[i]["dcam_tl"] = -1
                explanation[i]["permutation_success_tl"] = 0


            try:
                dcam_ol,permutation_success_ol = dcam.run(
                    instance=instance, nb_permutation=200, label_instance=output_label)
                explanation[i]["dcam_ol"] = dcam_ol
                explanation[i]["permutation_success_ol"] = permutation_success_ol
                plot(dcam_ol,instance,dataset_name,i,False)
            except IndexError:
                explanation[i]["dcam_ol"] = -1
                explanation[i]["permutation_success_ol"] = 0


            explanation[i]["ground_truth_label"] = gt_label
            explanation[i]["output_label"] = output_label
        np.save("dCCAM_results/"+dataset_name+"_explenations",explanation)

if __name__ == "__main__" :
    main()