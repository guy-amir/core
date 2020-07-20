import torch
from torch.utils.data import DataLoader, Dataset

def circle_gen(n_samples=1000,dim=1,radius=0.5):
    samples = dim*torch.rand(n_samples,2)-(dim/2)
    labels = ((samples**2).sum(1) <= radius**2)
    return samples,labels

class circlePointsDataset(Dataset):
    
  def __init__(self, samples, labels):
        'Initialization'
        self.labels = labels
        self.samples = samples

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

  def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.samples[index]
        y = self.labels[index]

        return X, y

Train_samples,labels = circle_gen(n_samples=100,dim=1,radius=0.5)
Valid_data,vlabels = circle_gen(n_samples=100,dim=1,radius=0.5)



train_dataset = circlePointsDataset(Train_samples,labels)
valid_dataset = circlePointsDataset(Valid_data,vlabels)

print(train_dataset)