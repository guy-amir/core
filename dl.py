import torchvision
import torch
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split

def get_dataloaders(prms):
    if prms.dataset == 'cifar10':
        return cifar_dl(prms)
    if prms.dataset == 'diabetes':
        return diabetes_dl(prms)

def diabetes_dl(prms):
    
    init_dataset = diabetesDataset(prms)
    diab_length = 768
    params = {'batch_size': diab_length,
          'shuffle': True,
          'num_workers': 6}
    
    lengths = [int(diab_length*0.8), diab_length-int(diab_length*0.8)]
    train_dataset,valid_dataset = random_split(init_dataset, lengths)

    train_loader = DataLoader(train_dataset, **params)
    test_loader = DataLoader(valid_dataset, **params)

    return train_dataset,valid_dataset,train_loader,test_loader

class diabetesDataset(Dataset):
    
    def __init__(self,prms):
        'Initialization'
        DATA_PATH = prms.data_path
        diab = pd.read_csv(f'{DATA_PATH}/diabetes.csv')
        
        X=diab[diab.columns[:8]]
        Y=diab['Outcome']

        self.samples = torch.tensor(X.values)
        self.labels = torch.tensor(Y.values)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.samples[index]
        y = self.labels[index]

        return X, y


def cifar_dl(prms):
    DATA_PATH = prms.data_path
    train_bs = prms.train_bs
    test_bs = prms.test_bs

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs,
                                            shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs,
                                            shuffle=False, num_workers=4)
    
    return trainset, testset, trainloader, testloader