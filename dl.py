import torchvision
import torch
import torchvision.transforms as transforms

def get_dataloaders(prms):
    if prms.dataset == 'cifar10':
        return cifar_dl(prms)


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
    
    return trainloader,testloader