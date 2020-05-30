import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
class cifar_net(nn.Module):
    def __init__(self):
        super(cifar_net, self).__init__()
        self.conv_layer1 = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.conv_layer2 = nn.Sequential(
            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
        )
        
        self.conv_layer3 = nn.Sequential(
            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer1 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        
        # self.fc_layer2 = nn.Sequential(
        #     nn.Linear(1024, 512),
            
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(512, 10)
        # )

        self.fc_layer2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256)
        )

        # self.for_tree = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(512, 10)
        # )

        # self.softmax = nn.Sequential(
        #     nn.Softmax() #dim=1) #maybe add dim if necessarry
        # )

    def forward(self, x):

        cl1 = self.conv_layer1(x)
        cl2 = self.conv_layer2(cl1)
        cl3 = self.conv_layer3(cl2)

        # flatten
        cl3 = cl3.view(cl3.size(0), -1)
        
        # fc layer
        fc1 = self.fc_layer1(cl3)
        fc2 = self.fc_layer2(fc1)

        #softmax
        # sm = self.softmax(fc2)

        # return x,cl1,cl2,cl3,fc1,fc2,sm #option a - smoothness testing
        return fc2 #option b - no smoothness testing

class Forest(nn.Module):
    def __init__(self, prms):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.prms = prms
        self.y_hat_avg= []
        self.mu_list = []



        #The neural network that feeds into the trees:
        self.prenet = cifar_net()

        for _ in range(self.prms.n_trees):
            tree = Tree(prms)
            self.trees.append(tree)

    def forward(self, xb,yb=None,layer=None):

        self.predictions = []
        if self.training:
            #convert yb from tensor to one_hot
            yb_onehot = torch.zeros(yb.size(0), int(yb.max()+1))
            yb = yb.view(-1,1)
            if yb.is_cuda:
                yb_onehot = yb_onehot.cuda()
            yb_onehot.scatter_(1, yb, 1)

            self.predictions = []            
            self.mu = []
            

        if self.prms.use_prenet:
            xb = self.prenet(xb)

        if (self.prms.use_tree == False):
            return xb

        for tree in self.trees: 
            
            #construct routing probability tree:
            mu = tree(xb)

            #find the nodes that are leaves:
            mu_midpoint = int(mu.size(1)/2)

            mu_leaves = mu[:,mu_midpoint:]
            # NL = mu_leaves.sum(1)
            #create a normalizing factor for leaves:
            N = mu.sum(0)
            

            if self.training:
                if self.prms.classification:
                    self.y_hat = yb_onehot.t() @ mu/N
                    y_hat_leaves = self.y_hat[:,mu_midpoint:]
                    self.y_hat_batch_avg.append(self.y_hat.unsqueeze(2))
            ####################################################################
            else: 
                y_hat_val_avg = torch.cat(self.y_hat_avg, dim=2)
                y_hat_val_avg = torch.sum(y_hat_val_avg, dim=2)/y_hat_val_avg.size(2)
                y_hat_leaves = y_hat_val_avg[:,mu_midpoint:]
            ####################################################################
            pred = (mu_leaves @ y_hat_leaves.t())

            self.predictions.append(pred.unsqueeze(1))
            

        ####################################################
        # if self.training:
        #     self.y_hat_batch_avg = torch.cat(self.y_hat_batch_avg, dim=2)
        #     self.y_hat_batch_avg = torch.sum(self.y_hat_batch_avg, dim=2)/self.prms.n_trees
        #     self.y_hat_avg.append(self.y_hat_batch_avg.unsqueeze(2))
        #######################################################

        self.prediction = torch.cat(self.predictions, dim=1)
        self.prediction = torch.sum(self.prediction, dim=1)/self.prms.n_trees
        return self.prediction

class Tree(nn.Module):
    def __init__(self,prms):
        super(Tree, self).__init__()
        self.depth = prms.tree_depth
        self.n_leaf = 2 ** prms.tree_depth
        self.n_nodes = self.n_leaf#-1
        self.n_features = prms.features4tree
        self.mu_cache = []
        self.prms = prms

        self.decision = nn.Sigmoid()

        #################################################################################################################
        onehot = np.eye(prms.feature_length)
        # randomly use some neurons in the feature layer to compute decision function
        self.using_idx = np.random.choice(prms.feature_length, self.n_leaf, replace=True)
        self.feature_mask = onehot[self.using_idx].T
        self.feature_mask = nn.parameter.Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor), requires_grad=False)
        #################################################################################################################


    def forward(self, x, save_flag = False):
        if x.is_cuda and not self.feature_mask.is_cuda:
            self.feature_mask = self.feature_mask.cuda()
        feats = torch.mm(x.view(-1,self.feature_mask.size(0)), self.feature_mask)
        decision = self.decision(feats) # passed sigmoid->[batch_size,n_leaf]

        decision = self.decision(feats) # passed sigmoid->[batch_size,n_leaf]

        decision = torch.unsqueeze(decision,dim=2) # ->[batch_size,n_leaf,1]
        decision_comp = 1-decision
        decision = torch.cat((decision,decision_comp),dim=2) # -> [batch_size,n_leaf,2]

        mu = x.data.new(x.size(0),1,1).fill_(1.)
        big_mu = x.data.new(x.size(0),2,1).fill_(1.)
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            # mu stores the probability a sample is routed at certain node
            # repeat it to be multiplied for left and right routing
            mu = mu.repeat(1, 1, 2)
            # the routing probability at n_layer
            _decision = decision[:, begin_idx:end_idx, :] # -> [batch_size,2**n_layer,2]
            mu = mu*_decision # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer+1)
            # merge left and right nodes to the same layer
            mu = mu.view(x.size(0), -1, 1)
            big_mu = torch.cat((big_mu,mu),1)

        big_mu = big_mu.view(x.size(0), -1)      
        return big_mu #-> [batch size,n_leaf]

def level2nodes(tree_level):
    return 2**(tree_level+1)

def level2node_delta(tree_level):
    start = level2nodes(tree_level-1)
    end = level2nodes(tree_level)
    return [start,end]