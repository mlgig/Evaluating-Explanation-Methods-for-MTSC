from load_data import *
from utilities import transform_data4ResNet
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
import timeit
from sklearn.metrics import accuracy_score

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
    plt.savefig("dCAM_results/plots/"+dataset_name+"/"+file_name)
    #plt.colorbar(img)

def main():
    all_data = load_data("synth")


    for dataset_name in all_data.keys():
        train_dataloader,test_dataloader, n_channels, n_classes, device,enc = transform_data4ResNet(all_data[dataset_name],dataset_name)
        modelarch = torch.load("saved_model/resNet/"+dataset_name+".ph")
        resnet = ModelCNN(model=modelarch ,n_epochs_stop=30,device=device)#,save_path='saved_model/resNet/'
        #+dataset_name+"_nFilters_"+str(mid_channels)+"_"+str(i))
        cnn_output = resnet.predict( test_dataloader )
        symbolic_output = enc.inverse_transform(cnn_output)
        print(dataset_name,"accuracy is",accuracy_score(symbolic_output,all_data[dataset_name]["y_test"]))
        # explain
        last_conv_layer = resnet.model._modules['layers'][2]
        fc_layer_name = resnet.model._modules['final']

        testSet_length =  all_data[dataset_name]["X_test"].shape[0]
        target_idxs = enc.transform(all_data[dataset_name]["y_test"])
        explanation= [ {} for i in range (testSet_length)]
        dcam = DCAM(resnet.model,device,last_conv_layer=last_conv_layer,fc_layer_name=fc_layer_name)

        starttime = timeit.default_timer()
        for i in range(testSet_length):
            #TODO modify the code for CMJ. Compute just the 6 possible permutations
            #TODO handle the case in which none of the permutations are according to the labels
            #TODO check the following warning
            # MatplotlibDeprecationWarning: Auto-removal of overlapping axes is deprecated since 3.6 and will be removed two minor releases later; explicitly call ax.remove() as needed.
            #   plt.subplot(nb_dim,1,1+i)
            print("explaining ",i,"-th sample of",dataset_name,"out of",testSet_length)

            instance = all_data[dataset_name]["X_test"][i] if type(all_data[dataset_name]["X_train"])==np.ndarray else  \
                all_data[dataset_name]["X_test"].values[i]
            gt_label = target_idxs[i]
            output_label = cnn_output[i]

            nb_permutation = 200
            try:
                dcam_tl,permutation_success_tl = dcam.run(
                    instance=instance, nb_permutation=nb_permutation, label_instance=gt_label)
                explanation[i]["dcam_tl"] = dcam_tl
                explanation[i]["permutation_success_tl"] = permutation_success_tl
                plot(dcam_tl,instance,dataset_name,i,True)
            except IndexError:
                explanation[i]["dcam_tl"] = -1
                explanation[i]["permutation_success_tl"] = 0
                sys.stderr.write("index error in ground truth""""""""\n\n")

            try:
                dcam_ol,permutation_success_ol = dcam.run(
                    instance=instance, nb_permutation=nb_permutation, label_instance=output_label)
                explanation[i]["dcam_ol"] = dcam_ol
                explanation[i]["permutation_success_ol"] = permutation_success_ol
                plot(dcam_ol,instance,dataset_name,i,False)
            except IndexError:
                explanation[i]["dcam_ol"] = -1
                explanation[i]["permutation_success_ol"] = 0
                sys.stderr.write("index error in predictions""""""""\n\n")


            explanation[i]["ground_truth_label"] = all_data[dataset_name]["y_test"][i]
            explanation[i]["output_label"] = symbolic_output[i]
        print("average time spent was", (timeit.default_timer() - starttime)/testSet_length)
        np.save("dCAM_results/"+dataset_name+"_explenations",explanation)

if __name__ == "__main__" :
    main()