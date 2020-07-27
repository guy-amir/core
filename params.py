import torch

class parameters():
    def __init__(self):

        #Computational parameters:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #Dataset parameters:
        self.dataset = 'cifar10'
        self.data_path = '../data'
        self.train_bs = 256
        self.test_bs = 1024

        #NN parameters:
        # self.batchnorm = True

        #Forest parameters:
        self.use_tree = True
        self.use_prenet = False
        self.classification = True

        self.n_trees = 2

        #Tree parameters:
        self.tree_depth = 6
        self.n_leaf = 2**self.tree_depth
        
        self.cascading = False
        self.single_level_training = True
        
        self.feature_map = False
        self.feature_length = 256
        self.features4tree = 2
        # self.level = 0

        # self.softmax_normalization = True ##! replace softmax_normalization in tree_conf

        self.logistic_regression_per_node = True

        #Training parameters:
        self.epochs = 2
        # self.batch_size = 64
        self.learning_rate = 0.01
        self.weight_decay=1e-5
        self.momentum=0.9

        #Wavelet parameters:
        self.wavelets = True
        self.intervals = 200

        #smoothness parameters:
        self.check_smoothness = False
