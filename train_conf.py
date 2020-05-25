import torch.optim as optim
import torch.nn as nn
import torch

class Trainer():
    def __init__(self,prms,net):

        self.prms = prms
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(net.parameters(), lr=prms.learning_rate, momentum=prms.momentum)

    def fit(self,trainloader):
        prms = self.prms

        for epoch in range(prms.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the x; data is a list of [x, y]
                x, y = data[0].to(prms.device), data[1].to(prms.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                preds = self.net(x)
                loss = self.criterion(preds, y)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

    def validation(self,testloader):
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