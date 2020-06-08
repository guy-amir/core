#load libraries
from params import parameters
from dl import get_dataloaders
from model_conf import Forest
from train_conf import Trainer
import pandas as pd

#load default parameters (including device)
prms = parameters()

#dataloaders
trainset, testset, trainloader, testloader = get_dataloaders(prms)

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

#load default parameters (including device)
# prms = parameters()

prms.use_tree = True
evaluate_network(prms)
prms.use_tree = False
evaluate_network(prms)




#postprocessing