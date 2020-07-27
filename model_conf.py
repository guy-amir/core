import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
class cifar_net(nn.Module):
    def __init__(self,prms=None):
        super(cifar_net, self).__init__()

        self.prms = prms

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

        if self.prms.check_smoothness:
            return x,cl1,cl2,cl3,fc1,fc2 #option a - smoothness testing
        else:
            return fc2 #option b - no smoothness testing

class Forest(nn.Module):
    def __init__(self, prms):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.prms = prms
        self.y_hat_avg= []
        self.mu_list = []

        #The neural network that feeds into the trees:
        self.prenet = cifar_net(self.prms)

        for _ in range(self.prms.n_trees):
            tree = Tree(prms)
            self.trees.append(tree)

    def forward(self, xb,yb=None,layer=None, save_flag = False):

        self.save_flag = save_flag
        self.predictions = []

        # if prms.use_pi == True:
        if self.training:
            #convert yb from tensor to one_hot
            yb_onehot = self.vec2onehot(yb)
        if self.prms.use_prenet:
            xb = self.prenet(xb)
    

        if (self.prms.use_tree == False):
            return xb

        for tree in self.trees: 
            
            mu = tree(xb)

            if self.training:
                self.predict(mu,yb_onehot)
            else:
                self.predict(mu)
            
        ##GG add averaging of trees 
        self.prediction = torch.cat(self.predictions, dim=1)
        self.prediction = torch.sum(self.prediction, dim=1)/self.prms.n_trees

        return self.prediction

    def predict(self,mu,yb_onehot=None):

        #find the nodes that are leaves:
        mu_midpoint = int(mu.size(1)/2)

        mu_leaves = mu[:,mu_midpoint:]

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

        if self.save_flag:
            self.mu_list.append(mu)
            self.y_hat_val_avg = y_hat_val_avg

        self.predictions.append(pred.unsqueeze(1))
    
    def vec2onehot(self,yb):
        yb_onehot = torch.zeros(yb.size(0), int(yb.max()+1))
        yb = yb.view(-1,1)
        if yb.is_cuda:
            yb_onehot = yb_onehot.cuda()
        yb_onehot.scatter_(1, yb, 1)
        return yb_onehot

    def forward_wavelets(self, xb,cutoff_nodes,yb=None,  layer=None, save_flag = False):

        #convert yb from tensor to one_hot
        yb_onehot = torch.zeros(yb.size(0), int(yb.max()+1))
        yb = yb.view(-1,1)
        if yb.is_cuda:
            yb_onehot = yb_onehot.cuda()
        yb_onehot.scatter_(1, yb, 1)

        self.predictions = []            
        

        if self.prms.use_prenet:
            xb = self.prenet(xb)

        if (self.prms.use_tree == False):
            return xb

        for tree in self.trees: 
            
            #construct routing probability tree:
            mu = tree(xb)

            nu = torch.zeros(mu.size())

            #find the nodes that are leaves:
            leaves = torch.zeros(mu.size(1))
            for j in cutoff_nodes:
                nu[:,j] = mu[:,j]
                if 2*j>=nu.size(1):
                    leaves[j] = 1
                else:
                    if not (cutoff_nodes==2*j).sum() and not (cutoff_nodes==(2*j+1)).sum():
                        leaves[j] = 1

            # print(f"leaves: {leaves}")

            #normalize leaf probabilities:
            nu_leaves = nu*leaves
            nu_normalize_factor = nu_leaves.sum(1)
            nu_normalized = (nu_leaves.t()/nu_normalize_factor).cuda()
            
            # N = mu.sum(0)
            
            eps = 10^-20

            self.y_hat = nu_normalized.cuda() @ yb_onehot
            self.y_hat = self.y_hat.t()/(self.y_hat.sum(1)+eps)

            pred = (self.y_hat @ nu_normalized.cuda()).t()

            if save_flag:
                self.mu_list.append(mu)
                self.y_hat_val_avg = y_hat_val_avg

            self.predictions.append(pred.unsqueeze(2))

        self.prediction = torch.cat(self.predictions, dim=2)
        self.prediction = torch.sum(self.prediction, dim=2)/self.prms.n_trees

        if self.prms.check_smoothness == True:
            self.pred_list = list(self.pred_list)
            self.pred_list.append(self.prediction)
            return self.pred_list
        else:
            return self.prediction

class Tree(nn.Module):
    def __init__(self,prms):
        super(Tree, self).__init__()
        self.depth = prms.tree_depth
        self.n_nodes = prms.n_leaf
        self.mu_cache = []
        self.prms = prms

        self.decision = nn.Sigmoid()

        if prms.feature_map == True:
            self.n_features = prms.feature_length
            onehot = np.eye(prms.feature_length)
            # randomly use some neurons in the feature layer to compute decision function
            self.using_idx = np.random.choice(prms.feature_length, prms.n_leaf, replace=True)
            self.feature_mask = onehot[self.using_idx].T
            self.feature_mask = nn.parameter.Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor), requires_grad=False)

        if prms.logistic_regression_per_node == True:
            self.fc = nn.ModuleList([nn.Linear(prms.n_leaf, 1).float() for i in range(self.n_nodes)])


    def forward(self, x, save_flag = False):
        if self.prms.feature_map == True:
            if x.is_cuda and not self.feature_mask.is_cuda:
                self.feature_mask = self.feature_mask.cuda()
            feats = torch.mm(x.view(-1,self.feature_mask.size(0)), self.feature_mask)
        else:
            feats = x

        self.d = [self.decision(node(feats)) for node in self.fc]
        
        self.d = torch.stack(self.d)

        decision = torch.cat((self.d,1-self.d),dim=2).permute(1,0,2)
        
        batch_size = x.size()[0]
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
        # self.mu_cache.append(big_mu)  
        return big_mu #-> [batch size,n_leaf]

def level2nodes(tree_level):
    return 2**(tree_level+1)

def level2node_delta(tree_level):
    start = level2nodes(tree_level-1)
    end = level2nodes(tree_level)
    return [start,end]