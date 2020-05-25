#load libraries
from params import parameters
from dl import get_dataloaders
from model_conf import cifar_net
from train_conf import Trainer

#load default parameters (including device)
prms = parameters()

#dataloaders
trainloader, testloader = get_dataloaders(prms)

#initiate model
net = cifar_net()

net.to(prms.device)

#run\fit\whatever
trainer = Trainer(prms,net)
trainer.fit(trainloader)
trainer.validation(testloader)
#postprocessing