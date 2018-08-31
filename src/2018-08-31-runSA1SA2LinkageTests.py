import ggcnn.experiment as experiment
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict


def load_sa1_dataset():
    keys_SA1 = []
    features_SA1 = []
    labels = []
    keys_SA2 = []
    features_SA2 = []
    
    # Load SA1 Node Features
    with open('Data/2018-08-24-NSW-SA1Input-Normalised.csv', 'r') as file:
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')  # Last value in line is \n
            keys_SA1.append(s[0])
            features_SA1.extend([float(v) for v in s[1:-1]])  # Last column is the outcome y
#             labels.append(np.floor(float(s[-1]) / 10).astype(int))
            labels.append(float(s[-1]))
    
    
    # Load SA2 Node Features
    with open('Data/2018-08-28-NSW-SA2Input-Normalised.csv', 'r') as file:
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
    with open('Data/2018-08-25-NSW-NeighbourDistance.csv', 'r') as file:
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
    with open('Data/Geography/2018-08-28-NSW-SA2_Neighbouring_Suburbs_With_Bridges-GCC.csv', 'r') as file:
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
    with open('Data/SA1SA2Links.csv', 'r') as file:
        adj_mat_SA1SA2 = np.zeros((len(keys_SA1), len(keys_SA2)))
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            a = keys_SA1.index(s[0])
            b = keys_SA2.index(s[1])
            adj_mat_SA1SA2[a, b] = 1
    
    adj_mat_SA2SA1 = np.transpose(adj_mat_SA1SA2)
    
    return features_SA1, adj_mat_SA1, labels, features_SA2, adj_mat_SA2, adj_mat_SA1SA2, adj_mat_SA2SA1

dataset = load_sa1_dataset()


class SA1Experiment():
    def __init__(self, neurons, blocks, reverseLinkagePosition = "Early", linkagePosition = "Late",
                    linkageActFun = True, linkageBatchNorm = True, linkageNeurons = None,
                    auxilaryEmbedding1 = False, auxilaryEmbedding2 = False,
                    auxilaryGraph = False):
        self.blocks = blocks
        self.neurons = neurons
        self.reverseLinkagePosition = reverseLinkagePosition
        self.linkagePosition = linkagePosition
        self.linkageActFun = linkageActFun
        self.linkageBatchNorm = linkageBatchNorm
        self.linkageNeurons = linkageNeurons
        self.auxilaryEmbedding1 = auxilaryEmbedding1
        self.auxilaryEmbedding2 = auxilaryEmbedding2
        self.auxilaryGraph = auxilaryGraph
    
    def create_network(self, net, input):
        net.create_network(input)
        
        if self.reverseLinkagePosition == "Early" or self.reverseLinkagePosition == "Both":
            net.make_reverse_auxilary_linkage_layer(self.linkageNeurons, with_act_func = self.linkageActFun, with_bn = self.linkageBatchNorm)
        if self.linkagePosition == "Early" or self.linkagePosition == "Both":
            net.make_auxilary_linkage_layer(self.linkageNeurons, with_act_func = self.linkageActFun, with_bn = self.linkageBatchNorm)
        
        net.make_embedding_layer(self.neurons)
        net.make_dropout_layer()
        
        for _ in range(self.blocks):
            net.make_graphcnn_layer(self.neurons)
            net.make_dropout_layer()
            net.make_embedding_layer(self.neurons)
            net.make_dropout_layer()
        
        if self.auxilaryEmbedding1:
            net.make_auxilary_embedding_layer(self.neurons)
            net.make_dropout_layer(input_type = 'current_V_auxilary')
        if self.reverseLinkagePosition == "Late" or self.reverseLinkagePosition == "Both":
            net.make_reverse_auxilary_linkage_layer(self.linkageNeurons, with_act_func = self.linkageActFun, with_bn = self.linkageBatchNorm)
        if self.auxilaryEmbedding2:
            net.make_auxilary_embedding_layer(self.neurons)
            net.make_dropout_layer(input_type = 'current_V_auxilary')
        if self.auxilaryGraph:
            net.make_auxilary_graphcnn_layer(self.neurons)
            net.make_dropout_layer(input_type = 'current_V_auxilary')
        if self.linkagePosition == "Late" or self.linkagePosition == "Both":
            net.make_auxilary_linkage_layer(self.linkageNeurons, with_act_func = self.linkageActFun, with_bn = self.linkageBatchNorm)
        
        net.make_embedding_layer(self.neurons)
        net.make_embedding_layer(1, name='final', with_bn=False, with_act_func = False)


######

def run(no_folds = 5, supervised = True, i = 0, l = 2, n = 128, expParameters = {}):
    inst = KFold(n_splits = no_folds, shuffle=True, random_state=125)
        
    exp = experiment.GGCNNExperiment('2018-08-28-SA1SA2', '2018-08-28-SA1SA2', SA1Experiment(neurons = n, blocks = l, **expParameters))

    exp.num_iterations = 100
    exp.optimizer = 'adam'
    exp.loss_type = 'linear'

    exp.debug = True  # Was True

    exp.preprocess_data(dataset)

    if supervised:
        train_idx, test_idx = list(inst.split(np.arange(len(dataset[0]))))[i]
    else:
        test_idx, train_idx = list(inst.split(np.arange(len(dataset[0]))))[i]  # Reversed to get more samples in the test set than the training set

    exp.create_data(train_idx, test_idx)
    exp.build_network()
    results = exp.run()

    return results


def runBatch(expParameters):
    l = 2
    n = 128

    # Collect results
    resultsDict = defaultdict(list)

    # Supervised
    no_folds = 5
    for i in range(no_folds):
        results = run(no_folds, True, i, l, n, expParameters)
        resultsDict['min_loss'].append(results[1][-1]['min loss'])
        resultsDict['i_vals'].append(i)
        resultsDict['supervised'].append(True)
        resultsDict['no_fold_vals'].append(no_folds)
        
    
    # Semi-supervised
    for i in range(no_folds):
        results = run(no_folds, False, i, l, n, expParameters)
        resultsDict['min_loss'].append(results[1][-1]['min loss'])
        resultsDict['i_vals'].append(i)
        resultsDict['supervised'].append(False)
        resultsDict['no_fold_vals'].append(no_folds)

    no_folds = 10
    for i in range(no_folds):
        results = run(no_folds, False, i, l, n, expParameters)
        resultsDict['min_loss'].append(results[1][-1]['min loss'])
        resultsDict['i_vals'].append(i)
        resultsDict['supervised'].append(False)
        resultsDict['no_fold_vals'].append(no_folds)

    numberOfResults = len(resultsDict['min_loss'])
    otherParams = dict( [(k, [v] * numberOfResults) for k,v in expParameters.items()])
    otherParamsDf = pd.DataFrame(otherParams)
    df = pd.DataFrame(resultsDict)
    df = pd.concat([df, otherParamsDf], axis = 1)

    return df


dfs = []

### Test 1
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "None", "linkageActFun": True, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": False}
dfs.append( runBatch(expParameters = expParameters) )

### Test 2
expParameters = {"reverseLinkagePosition": "Early", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True}
dfs.append( runBatch(expParameters = expParameters) )



### Combine and output
df = pd.concat(dfs)
df.to_csv("2018-08-31-TestResults.csv")
