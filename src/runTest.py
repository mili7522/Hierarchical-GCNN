import ggcnn.experiment as experiment

import numpy as np
import sys
import pandas as pd

from sklearn.model_selection import KFold

dataFolder = '../../../Graph-CNN/Graph-CNN/src/'

def load_dataset():
    keys = []
    features = []
    labels = []
    with open(dataFolder + 'Data/2018-06-03-SYD-SA1Input-Normalised.csv', 'r') as file:
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')  # Last value in line is \n
            keys.append(s[0])
            features.extend([float(v) for v in s[1:-1]])  # Last column is the outcome y
            labels.append(int(s[-1]))
        labels = np.array(labels)
        features = np.array(features).reshape((len(keys), -1))
    
    with open(dataFolder + 'Data/2018-06-03-SYD-NeighbourLinkFeatures.csv', 'r') as file:
        adj_mat = np.zeros((len(labels), 1, len(labels)))
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            a = keys.index(s[0])
            b = keys.index(s[1])
            adj_mat[a, 0, b] = 1
            adj_mat[b, 0, a] = 1
    return features, adj_mat, labels

dataset = load_dataset()

class SA1Experiment():
    def __init__(self, neurons, blocks):
        self.blocks = blocks
        self.neurons = neurons
    
    def create_network(self, net, input):
        net.create_network(input)
        net.make_adjacency_adjustment_layer()
        net.make_embedding_layer(self.neurons)
        net.make_dropout_layer()
        
        for _ in range(self.blocks):
            net.make_graphcnn_layer(self.neurons)
            net.make_dropout_layer()
            net.make_embedding_layer(self.neurons)
            net.make_dropout_layer()
        
        net.make_embedding_layer(10, name='final', with_bn=False, with_act_func = False)


no_folds = 5
inst = KFold(n_splits = no_folds, shuffle=True, random_state=125)

try:
    l = int(sys.argv[1])
    n = int(sys.argv[2])
    i = int(sys.argv[3])
except IndexError:
    l = 2
    n = 128
    i = 0
    
saveName = 'Output/SemiSupervisedSydney-NoEdge-l={:d}-n={:d}-i={:d}.csv'.format(l,n,i)

max_acc = []
iteration = []
layers = []
neurons = []
rep = []

    
exp = experiment.GGCNNExperiment('2018-06-06-SA1', '2018-06-06-sa1', SA1Experiment(neurons = n, blocks = l))
# exp = experiment.SingleGraphCNNExperiment('2018-06-06-SA1', '2018-06-06-sa1', SA1Experiment(neurons = n, blocks = l))

exp.num_iterations = 1000
exp.optimizer = 'adam'

exp.debug = True  # Was True

exp.preprocess_data(dataset)

# train_idx, test_idx = list(inst.split(np.arange(len(dataset[-1]))))[i]
test_idx, train_idx = list(inst.split(np.arange(len(dataset[-1]))))[i]  # Reversed to get more samples in the test set than the training set


exp.create_data(train_idx, test_idx)
exp.build_network()
results = exp.run()

# exp.build_network()
# results = exp.run()