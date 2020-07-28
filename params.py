import torch

class parameters():
    def __init__(self):

        #Computational parameters:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #Dataset parameters:
        self.dataset = 'cifar10' #"mnist" #'wine'
        self.data_path = '../data'
        self.train_bs = 256
        self.test_bs = 1024

        #NN parameters:
        # self.batchnorm = True

        #Forest parameters:
        self.use_tree = True
        self.use_prenet = True
        self.classification = True
        self.use_pi = True

        self.n_trees = 1

        #Tree parameters:
        self.tree_depth = 4
        self.n_leaf = 2**self.tree_depth
        if self.dataset == 'cifar10':
            self.feature_length = 256
            self.n_classes = 10
        self.cascading = False
        self.single_level_training = True
        self.features4tree = 1
        self.logistic_regression_per_node = True
        self.feature_map = True
        

        #Training parameters:
        self.epochs = 20
        # self.batch_size = 64
        self.learning_rate = 0.001
        self.weight_decay=1e-4
        self.momentum=0.9

        #Wavelet parameters:
        self.wavelets = False
        self.intervals = 200

        #smoothness parameters:
        self.check_smoothness = False
