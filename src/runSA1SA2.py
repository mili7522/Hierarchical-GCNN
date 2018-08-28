# %load runSA1SA2.py
import ggcnn.experiment as experiment
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

SA1DatasetSize = 0
dataFolder = ''

def load_sa1_dataset():
    global SA1DatasetSize
    keys_SA1 = []
    features_SA1 = []
    labels = []
    keys_SA2 = []
    features_SA2 = []
    # Load SA1 Node Features
    with open(dataFolder + 'Data/2018-08-24-NSW-SA1Input-Normalised.csv', 'r') as file:
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')  # Last value in line is \n
            keys_SA1.append(s[0])
            features_SA1.extend([float(v) for v in s[1:-1]])  # Last column is the outcome y
#             labels.append(np.floor(float(s[-1]) / 10).astype(int))
            labels.append(float(s[-1]))
    
    SA1DatasetSize = len(labels)
    
    # Load SA2 Node Features
    with open(dataFolder + 'Data/2018-08-28-NSW-SA2Input-Normalised.csv', 'r') as file:
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')  # Last value in line is \n
            keys_SA2.append(s[0])
            features_SA2.extend([float(v) for v in s[1:-1]])  # Last column is the outcome y

    labels = np.array(labels)
    features_SA1 = np.array(features_SA1).reshape((len(keys_SA1), -1))
    features_SA2 = np.array(features_SA2).reshape((len(keys_SA2), -1))
    
    # Load SA1 Link Features
    with open(dataFolder + 'Data/2018-08-25-NSW-NeighbourDistance.csv', 'r') as file:
        adj_mat_SA1 = np.zeros((len(keys_SA1), len(keys_SA1)))
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            a = keys_SA1.index(s[0])
            b = keys_SA1.index(s[1])
            adj_mat_SA1[a, b] = 1
            adj_mat_SA1[b, a] = 1

    # Load SA2 Link Features
    with open(dataFolder + 'Data/Geography/2018-08-28-NSW-SA2_Neighbouring_Suburbs_With_Bridges-GCC.csv', 'r') as file:
        adj_mat_SA2 = np.zeros((len(keys_SA2), len(keys_SA2)))
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            a = keys_SA2.index(s[0])
            b = keys_SA2.index(s[1])
            adj_mat_SA2[a, b] = 1
            adj_mat_SA2[b, a] = 1
    
    # Load SA1, SA2 Links
    with open(dataFolder + 'Data/SA1SA2Links.csv', 'r') as file:
        adj_mat_SA1SA2 = np.zeros((len(keys_SA1), len(keys_SA2)))
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            a = keys_SA1.index(s[0])
            b = keys_SA2.index(s[1])
            adj_mat_SA1SA2[a, b] = 1

    
    return features_SA1, adj_mat_SA1, labels, features_SA2, adj_mat_SA2, adj_mat_SA1SA2

dataset = load_sa1_dataset()

class SA1Experiment():
    def __init__(self, neurons, blocks):
        self.blocks = blocks
        self.neurons = neurons
    
    def create_network(self, net, input):
        net.create_network(input)
        net.make_embedding_layer(self.neurons)
        net.make_dropout_layer()
        
        for _ in range(self.blocks):
            net.make_graphcnn_layer(self.neurons)
            net.make_dropout_layer()
            net.make_embedding_layer(self.neurons)
            net.make_dropout_layer()
        
        net.make_auxilary_graphcnn_layer(self.neurons)
        net.make_auxilary_linkage_layer(self.neurons)
        
        net.make_embedding_layer(self.neurons)
        net.make_graphcnn_layer(1, name='final', with_bn=False, with_act_func = False)


no_folds = 5 ##
inst = KFold(n_splits = no_folds, shuffle=True, random_state=125)


l = 2
n = 64
i = 2


exp = experiment.GGCNNExperiment('2018-08-28-SA1SA2', '2018-08-28-SA1SA2', SA1Experiment(neurons = n, blocks = l))

exp.num_iterations = 2000
exp.optimizer = 'adam'
exp.loss_type = "linear"

exp.debug = True  # Was True

exp.preprocess_data(dataset)

train_idx, test_idx = list(inst.split(np.arange( len(dataset[0]) )))[i]
# print('Before: ', exp.train_idx.shape)
# exp.train_idx = np.append(exp.train_idx, np.arange( SA1DatasetSize , len(dataset[-1] )))
# exp.test_idx = np.append(exp.test_idx, np.arange( SA1DatasetSize , len(dataset[-1] )))
# print('After: ', exp.train_idx.shape)
# test_idx, train_idx = list(inst.split(np.arange(len(dataset[-1]))))[i]  # Reversed to get more samples in the test set than the training set


exp.create_data(train_idx, test_idx)
exp.build_network()
results = exp.run()
