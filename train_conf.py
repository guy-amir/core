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

            self.validation(testloader)
            
            # self.net.y_hat_avg = torch.cat(self.net.y_hat_avg, dim=2)
            # self.net.y_hat_avg = torch.sum(self.net.y_hat_avg, dim=2)/self.net.y_hat_avg.size(2)

