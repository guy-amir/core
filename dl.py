import torch
from torch.utils.data import DataLoader, Dataset

def circle_gen(n_samples=1000,dim=1,radius=0.5):
    samples = dim*torch.rand(n_samples,2)-(dim/2)
    return samples

class oversampdata(Dataset):
    
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        target = self.data[index][-1]
        data_val = self.data[index] [:-1]
        return data_val,target

Train_data = circle_gen(n_samples=100,dim=1,radius=0.5)
Valid_data = circle_gen(n_samples=100,dim=1,radius=0.5)

print(Train_data.size())

train_dataset = oversampdata(Train_data)
valid_dataset = oversampdata(Valid_data)