import torch
from torch.utils.data import DataLoader, Dataset, random_split
# torch.utils.data.random_split(dataset, lengths)

def get_dataloaders():
    params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
    
    init_dataset = circlePointsDataset(n_samples=1000, dim=1, radius=0.5)
    lengths = [int(len(init_dataset)*0.8), int(len(init_dataset)*0.2)]
    train_dataset,valid_dataset = random_split(init_dataset, lengths)

    train_loader = DataLoader(train_dataset, **params)
    test_loader = DataLoader(valid_dataset, **params)

    return train_loader,test_loader

class circlePointsDataset(Dataset):
    
    def __init__(self, n_samples=1000, dim=1, radius=0.5):
        'Initialization'
        samples,labels = self.circle_gen(n_samples,dim,radius)
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
    
    def circle_gen(self,n_samples=1000,dim=1,radius=0.5):
        samples = dim*torch.rand(n_samples,2)-(dim/2)
        labels = ((samples**2).sum(1) <= radius**2).int()
        return samples,labels

# Train_samples,labels = circle_gen(n_samples=100,dim=1,radius=0.5)
# Valid_data,vlabels = circle_gen(n_samples=100,dim=1,radius=0.5)



# train_dataset = circlePointsDataset(n_samples=1000, dim=1, radius=0.5)
# valid_dataset = circlePointsDataset(n_samples=1000, dim=1, radius=0.5)

# print(train_dataset)