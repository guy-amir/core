import torch.optim as optim
import torch.nn as nn
import torch
from smooth import smoothness_layers

class Trainer():
    def __init__(self,prms,net):
        self.prms = prms
        self.net = net
        if prms.use_tree == True:
            self.criterion = nn.NLLLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        if prms.optimizer == 'SGD':
            self.optimizer = optim.SGD(net.parameters(), lr=prms.learning_rate, momentum=prms.momentum, weight_decay=self.prms.weight_decay)
        if prms.optimizer == 'Adam':
            self.optimizer = optim.Adam(net.parameters(), lr=prms.learning_rate, weight_decay=self.prms.weight_decay)

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
                acc = 100 * correct / total

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            acc))
        return acc

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
                if self.prms.check_smoothness == True:
                    preds = preds[-1]
                _, predicted = torch.max(preds, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                acc = 100 * correct / total

        print(f'Accuracy of the network with {cutoff} wavelets on the 10000 test images: {acc}')
        return acc

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

        self.loss_list = []
        self.val_acc_list = []
        self.train_acc_list = []
        self.wav_acc_list = []
        self.smooth_list = []
        self.cutoff_list = []

        for epoch in range(prms.epochs):  # loop over the dataset multiple times

            self.net.train(True)
            #add if for tree:
            
            self.net.y_hat_batch_avg = []

            total = 0
            correct = 0
            running_loss = 0.0
            long_running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the x; data is a list of [x, y]
                xb, yb = data[0].to(prms.device), data[1].to(prms.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                
                preds = self.net(xb,yb)
                if prms.use_tree==True:
                    loss = self.criterion(torch.log(preds), yb)
                else:
                    loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                long_running_loss  += loss.item()
                if i % 50 == 49:    # print every 50 mini-batches

                    print(f'[{epoch + 1}, {i + 1}] loss: {running_loss}')
                    running_loss = 0.0

            
            _, predicted = torch.max(preds, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
            train_acc = 100 * correct / total

            if prms.check_smoothness:
                preds_list = self.net.pred_list
                smooth_layers = smoothness_layers(preds_list,yb)


            if prms.use_tree:
                if prms.use_pi:
                    for tree in self.net.trees:
                        tree.pi_counter = nn.functional.softmax(tree.pi_counter, dim=1).data #GG??
                        tree.pi = nn.Parameter(tree.pi_counter, requires_grad = False)
                        tree.pi_counter = tree.pi.data.new(self.prms.n_leaf, self.prms.n_classes).fill_(.0)
                        print(f"pi: {tree.pi.sum()}")

                else:
                    wav_acc = []
                    self.net.y_hat_batch_avg = torch.cat(self.net.y_hat_batch_avg, dim=2)
                    self.net.y_hat_batch_avg = torch.sum(self.net.y_hat_batch_avg, dim=2)/self.net.y_hat_batch_avg.size(2)
                    self.net.y_hat_avg.append(self.net.y_hat_batch_avg.unsqueeze(2))
                    if prms.wavelets == True:
                        for i in range(1,6):
                            cutoff = int(i*prms.n_leaf/5) #arbitrary cutoff
                            wav_acc.append(self.wavelet_validation(testloader,cutoff))
            

            self.loss_list.append(long_running_loss)
            val_acc = self.validation(testloader)
            self.val_acc_list.append(val_acc)
            self.train_acc_list.append(train_acc)
            if prms.use_tree and prms.wavelets:
                self.wav_acc_list.append(wav_acc)
            self.cutoff_list = [int(i*prms.n_leaf/5) for i in range(1,6)]
            if prms.check_smoothness == True:
                self.smooth_list.append(smooth_layers)

            

        return self.loss_list,self.val_acc_list,self.train_acc_list,self.wav_acc_list,self.cutoff_list,self.smooth_list
            
            #this is where we append everything
