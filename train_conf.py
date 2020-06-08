import torch.optim as optim
import torch.nn as nn
import torch

class Trainer():
    def __init__(self,prms,net):
        self.prms = prms
        self.net = net
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(net.parameters(), lr=prms.learning_rate, momentum=prms.momentum, weight_decay=self.prms.weight_decay)

    def validation(self,testloader):
        self.net.train(False)
        prms = self.prms
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(prms.device), data[1].to(prms.device)
                preds = self.net(images)
                _, predicted = torch.max(preds, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    def wavelet_validation(self,testloader,cutoff):
        self.net.train(False)
        prms = self.prms
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(prms.device), data[1].to(prms.device)
                preds = self.net(images, save_flag=True)

                _, predicted = torch.max(preds, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            #this is where the magic happens:
            # 1. Calcuate phi:
            y = self.net.y_hat_val_avg #just create a shorthand to save typing a long name
            mu = self.net.mu_list #just create a shorthand to save typing a long name
            fixed_mu = [m for m in mu if m.size(0)==1024] #remove all the mus with less than 1024 samples
            mu = sum(fixed_mu)/(len(fixed_mu))
            mu = mu.mean(0)

            phi,phi_norm,sorted_nodes = self.phi_maker(y,mu)

            # 3. cutoff and add parents
            cutoff_nodes = sorted_nodes[:cutoff]

            for node in cutoff_nodes:

                for parent in self.find_parents(node.item()):

                    mask = (cutoff_nodes == parent.cpu())

                    if mask.sum() == 0:
                        cutoff_nodes = cutoff_nodes.tolist()
                        cutoff_nodes.append(parent.item())
                        cutoff_nodes = torch.LongTensor(cutoff_nodes)

            # 5. calculate values in new tree
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data[0].to(prms.device), data[1].to(prms.device)
                preds = self.net.forward_wavelets(xb = images, yb = labels, cutoff_nodes=cutoff_nodes)

                _, predicted = torch.max(preds, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network with {cutoff} wavelets on the 10000 test images: {100 * correct / total}')

    def phi_maker(self,y,mu):
        phi = torch.zeros(y.size())
        phi_norm = torch.zeros(y.size(1))
        #calculate the phis and the norms:
        for i in range(2,y.size(1)):
            p = self.find_parents(i)[0]
            phi[:,i] = mu[i]*(y[:,i]-y[:,p])
            phi_norm[i] = phi[:,i].norm(2)
        #Order phis from large to small:
        _,sorted_nodes = torch.sort(-phi_norm)
        return phi,phi_norm,sorted_nodes

    def find_parents(self,N):
        parent_list = []
        current_parent = N//2
        while(current_parent is not 0):
            parent_list.append(current_parent)
            current_parent = current_parent//2
        return torch.LongTensor(parent_list).cuda()

    def fit(self,trainloader,testloader):
        self.net.train(True)
        prms = self.prms
        self.net.y_hat_avg = []

        for epoch in range(prms.epochs):  # loop over the dataset multiple times

            self.net.train(True)
            #add if for tree:
            
            self.net.y_hat_batch_avg = []

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the x; data is a list of [x, y]
                xb, yb = data[0].to(prms.device), data[1].to(prms.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                
                preds = self.net(xb,yb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:    # print every 200 mini-batches
                    print(f'[{epoch + 1}, {i + 1}] loss: {running_loss}')
                    running_loss = 0.0

            self.net.y_hat_batch_avg = torch.cat(self.net.y_hat_batch_avg, dim=2)
            self.net.y_hat_batch_avg = torch.sum(self.net.y_hat_batch_avg, dim=2)/self.net.y_hat_batch_avg.size(2)
            self.net.y_hat_avg.append(self.net.y_hat_batch_avg.unsqueeze(2))

            for i in range(5):
                cutoff = int(i*prms.n_leaf/5) #arbitrary cutoff
                self.wavelet_validation(testloader,cutoff)
            self.validation(testloader)