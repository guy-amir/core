import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

class LitCIFAR(LightningModule):

  def __init__(self):
    super().__init__()

    # CIFAR images are (3, 32, 32) (channels, width, height)
    self.layer_1 = torch.nn.Linear(3 * 32 * 32, 128)
    self.layer_2 = torch.nn.Linear(128, 256)
    self.layer_3 = torch.nn.Linear(256, 10)

  def forward(self, x):
    batch_size, channels, width, height = x.size()

    # (b, 1, 28, 28) -> (b, 1*28*28)
    x = x.view(batch_size, -1)

    # layer 1
    x = self.layer_1(x)
    x = torch.relu(x)

    # layer 2
    x = self.layer_2(x)
    x = torch.relu(x)

    # layer 3
    x = self.layer_3(x)

    # probability distribution over labels
    x = torch.log_softmax(x, dim=1)

    return x

    def train_dataloader(self):


        transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                            download=True, transform=transform)

        return DataLoader(trainset, batch_size=64)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

net = LitCIFAR()
x = torch.Tensor(1, 3, 32, 32)
out = net(x)