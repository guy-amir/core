#load libraries
from params import parameters
from dl import get_dataloaders
from model_conf import Forest
from train_conf import Trainer

#load default parameters (including device)
prms = parameters()

#dataloaders
trainset, testset, trainloader, testloader = get_dataloaders(prms)

#initiate model
net = Forest(prms)

#move model to CUDA
net.to(prms.device)

#run\fit\whatever
trainer = Trainer(prms,net)
trainer.fit(trainloader,testloader)

#postprocessing