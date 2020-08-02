import torch

class parameters():
    def __init__(self):

        #Computational parameters:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #Dataset parameters:
        self.dataset = 'diabetes' #'cifar10' #"mnist" #'wine'
        self.data_path = '../data'
        self.train_bs = 256
        self.test_bs = 1024
        if self.dataset == 'cifar10':
            self.feature_length = 256
            self.n_classes = 10
        if self.dataset == 'diabetes':
            self.feature_length = 8
            self.n_classes = 2


        #NN parameters:
        # self.batchnorm = True

        #Forest parameters:
        self.use_tree = True
        self.use_prenet = False
        self.classification = True
        self.use_pi = True

        self.n_trees = 4

        #Tree parameters:
        self.tree_depth = 4
        self.n_leaf = 2**self.tree_depth


        self.cascading = False
        self.single_level_training = True
        self.features4tree = 1
        self.logistic_regression_per_node = True
        self.feature_map = True
        

        #Training parameters:
        self.epochs = 100
        # self.batch_size = 64
        self.learning_rate = 0.03
        self.weight_decay=1e-4
        self.momentum=0.9
        self.optimizer = 'Adam'

        #Wavelet parameters:
        self.wavelets = False
        self.intervals = 200

        #smoothness parameters:
        self.check_smoothness = False
