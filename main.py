#load libraries
from params import parameters
from dl import get_dataloaders
from model_conf import Forest
from train_conf import Trainer
import pandas as pd

#load default parameters (including device)
prms = parameters()

#dataloaders
trainloader, testloader = get_dataloaders()
print("hi")

# def df_maker(loss_list,val_acc_list,train_acc_list,wav_acc_list,cutoff_list,smooth_list):
#     df = pd.DataFrame({'loss_list':loss_list,'val_acc_list':val_acc_list,'train_acc_list':train_acc_list})
#     if prms.wavelets and prms.use_tree:
#         for ii in range(len(wav_acc_list[0])):
#             df[f'{cutoff_list[ii]} wavelets'] = [wav_acc_list[jj][ii] for jj in range(len(wav_acc_list))]
#     for kk in range(len(smooth_list[0])):
#         df[f'layer chunk {kk}'] = [smooth_list[jj][kk] for jj in range(len(smooth_list))]

#     return df

def evaluate_network(prms):
    #dataloaders
    # trainset, testset, trainloader, testloader = get_dataloaders(prms)

    ###here we will add a loop that will hange dataset size

    #initiate model:
    net = Forest(prms)
    net.to(prms.device) #move model to CUDA

    #run\fit\whatever
    trainer = Trainer(prms,net)
    loss_list,val_acc_list,train_acc_list,wav_acc_list,cutoff_list,smooth_list = trainer.fit(trainloader,testloader)
    print("hi")
    # df = df_maker(loss_list,val_acc_list,train_acc_list,wav_acc_list,cutoff_list,smooth_list)
    # return df

#load default parameters (including device)
# prms = parameters()

# lrs = [0.01]
# depths = [5,12]
# for lr in lrs:
#     prms.learning_rate = lr
#     prms.use_tree = True
#     for d in depths:
#         prms.depth = d
#         df = evaluate_network(prms)
#         df.to_csv(f'tree{prms.use_tree}lr{prms.learning_rate}depth{prms.depth}.csv')

#     prms.use_tree = False
#     df = evaluate_network(prms)
#     df.to_csv(f'tree{prms.use_tree}lr{prms.learning_rate}.csv')

prms.use_tree = True
prms.wavelets = False
df = evaluate_network(prms)
df.to_csv(f'epochs{prms.epochs}tree{prms.use_tree}lr{prms.learning_rate}depth{prms.tree_depth}.csv')



#postprocessing