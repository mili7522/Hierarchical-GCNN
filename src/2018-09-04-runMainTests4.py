import ggcnn.experiment as experiment
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import sklearn
from collections import defaultdict
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_sa1_dataset():
    keys_SA1 = []
    features_SA1 = []
    labels = []
    keys_SA2 = []
    features_SA2 = []
    
    # Load SA1 Node Features
    with open('Data/2018-09-02-NSW-SA1Input-Normalised.csv', 'r') as file:
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
    with open('Data/Geography/2018-09-01-NSW-Neighbouring_Suburbs_With_Bridges-Filtered.csv', 'r') as file:
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
    with open('Data/Geography/2018-09-01-SA2Neighbouring_Suburbs_With_Bridges-Filtered.csv', 'r') as file:
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
    with open('Data/2018-09-02-SA1SA2Links.csv', 'r') as file:
        adj_mat_SA1SA2 = np.zeros((len(keys_SA1), len(keys_SA2)))
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            a = keys_SA1.index(s[0])
            b = keys_SA2.index(s[1])
            adj_mat_SA1SA2[a, b] = 1
    
    adj_mat_SA2SA1 = np.transpose(adj_mat_SA1SA2)
    
    return (features_SA1, adj_mat_SA1, labels, features_SA2, adj_mat_SA2, adj_mat_SA1SA2, adj_mat_SA2SA1), (keys_SA1, keys_SA2)

dataset, keys = load_sa1_dataset()

####

class SA1Experiment():
    def __init__(self, neurons, blocks, reverseLinkagePosition = "Early", linkagePosition = "Late",
                    linkageActFun = True, linkageBatchNorm = True, linkageNeurons = None,
                    auxilaryEmbedding1 = False, auxilaryEmbedding2 = False,
                    auxilaryGraph = False, linkage_adjustment_components = None, reverse_linkage_adjustment_components = None,
                    linkage_W_2D = False):
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
        self.linkage_adjustment_components = linkage_adjustment_components
        self.reverse_linkage_adjustment_components = reverse_linkage_adjustment_components
        self.linkage_W_2D = linkage_W_2D
    
    def create_network(self, net, input):
        net.create_network(input)

        if self.linkage_adjustment_components is not None:
            net.make_linkage_adjustment_layer(twoD_W = self.linkage_W_2D)
        if self.reverse_linkage_adjustment_components is not None:
            net.make_reverse_linkage_adjustment_layer(twoD_W = self.linkage_W_2D)
        
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

    exp.num_iterations = 5000
    exp.optimizer = 'adam'
    exp.loss_type = 'linear'

    exp.debug = True  # Was True

    exp.preprocess_data(dataset)

    valid_idx = np.flatnonzero(dataset[2] >= 0)  # Missing data labelled with -1
    if supervised:
        train_idx, test_idx = list(inst.split( valid_idx ))[i]
    else:
        test_idx, train_idx = list(inst.split( valid_idx ))[i]  # Reversed to get more samples in the test set than the training set

    n_components = expParameters.get('linkage_adjustment_components', None)
    exp.create_data(train_idx, test_idx, n_components = n_components)
    exp.build_network()
    results = exp.run()
    
    # Node type of input nodes: 0 = training set; 1 = test set; -1 = neither
    idx_split = np.empty((len(dataset[2]), 1))
    idx_split.fill(-1)
    idx_split[train_idx] = 0
    idx_split[test_idx] = 1

    return results, idx_split


def appendResults(resultsDict, results, r2, corr, i, supervised, no_folds):
    resultsDict['min_loss'].append(results[1][-1]['min loss'])
    resultsDict['loss'].append(results[1][-1]['loss'])
    resultsDict['min train loss'].append(results[0][-1]['min loss'])
    resultsDict['r2'].append(r2)
    resultsDict['corr'].append(corr)
    resultsDict['i_vals'].append(i)
    resultsDict['supervised'].append(supervised)
    resultsDict['no_fold_vals'].append(no_folds)
    return resultsDict

def getPrediction(idx_split, results):
    test_idx = np.flatnonzero(idx_split == 1)
    predictions = results[-1].ravel()[test_idx]
    actual = dataset[2].ravel()[test_idx]

    predictions_df = pd.DataFrame(np.hstack((results[-1], dataset[2].reshape((-1,1)), idx_split)), columns=['Prediction', 'Actual', 'idx_split'])
    predictions_df.index = np.array(keys[0], dtype = int)

    r2 = sklearn.metrics.r2_score(predictions, actual)
    corr = np.corrcoef(predictions, actual)[1,0]

    return predictions_df, r2, corr

def saveResults(results, predictions_df, resultsFolder, exp_name, rep):
    for subfolder in ['Predictions', 'TrainFull', 'TestFull']:
        os.makedirs(os.path.join(resultsFolder, subfolder), exist_ok=True)

    train_df = pd.DataFrame(results[0])
    test_df = pd.DataFrame(results[1])
    test_df.set_index(test_df.index * 5, inplace = True)  # 5 is the interval between testing
    train_df.to_csv(os.path.join(resultsFolder, 'TrainFull', exp_name + rep + '_TrainDf.csv'))
    test_df.to_csv(os.path.join(resultsFolder, 'TestFull', exp_name + rep + '_TestDf.csv'))

    predictions_df.to_csv(os.path.join(resultsFolder, 'Predictions', exp_name + rep + '.csv'))

def runBatch(expParameters, l = 2, n = 128, exp_name = None):

    # Folders for saving results:
    resultsFolder = 'Results/'

    # Collect results
    resultsDict = defaultdict(list)

    # Supervised
    no_folds = 5
    supervised = True
    for i in range(no_folds):
        results, idx_split = run(no_folds, supervised, i, l, n, expParameters)
        predictions_df, r2, corr = getPrediction(idx_split, results)
        resultsDict = appendResults(resultsDict, results, r2, corr, i, supervised, no_folds)
        saveResults(results, predictions_df, resultsFolder, exp_name, "_Supervised-5-{}".format(i))
        
    # Semi-supervised
    supervised = False
    for i in range(no_folds):
        results, idx_split = run(no_folds, supervised, i, l, n, expParameters)
        predictions_df, r2, corr = getPrediction(idx_split, results)
        resultsDict = appendResults(resultsDict, results, r2, corr, i, supervised, no_folds)
        saveResults(results, predictions_df, resultsFolder, exp_name, "_Semisupervised-5-{}".format(i))

    no_folds = 10
    for i in range(no_folds):
        results, idx_split = run(no_folds, supervised, i, l, n, expParameters)
        predictions_df, r2, corr = getPrediction(idx_split, results)
        resultsDict = appendResults(resultsDict, results, r2, corr, i, supervised, no_folds)
        saveResults(results, predictions_df, resultsFolder, exp_name, "_Semisupervised-10-{}".format(i))

    numberOfResults = len(resultsDict['min_loss'])
    otherParams = dict( [(k, [v] * numberOfResults) for k,v in expParameters.items()])
    otherParamsDf = pd.DataFrame(otherParams)
    l_and_n_df = pd.DataFrame({'l': [l] * numberOfResults, 'n': [n] * numberOfResults})
    df = pd.DataFrame(resultsDict)
    df = pd.concat([df, otherParamsDf, l_and_n_df], axis = 1)

    return df


dfs = []
dfSaveName = "Results/2018-09-04-MainTests4.csv"

### Test 10
exp_number = 10
expParameters = {"reverseLinkagePosition": "Late", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None, "reverse_linkage_adjustment_components": None, "linkage_W_2D": False}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-04_EXP10_LateLateGC")
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Test 11
exp_number = 11
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": True, "auxilaryGraph": True,
                 "linkage_adjustment_components": None, "reverse_linkage_adjustment_components": None, "linkage_W_2D": False}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-04_EXP11_NoneLateEmbGC")
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Test 12
exp_number = 12
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": True, "auxilaryGraph": True,
                 "linkage_adjustment_components": None, "reverse_linkage_adjustment_components": None, "linkage_W_2D": False}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-04_EXP12_NoneLateEmbGC")
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)

### Test 13
exp_number = 10
expParameters = {"reverseLinkagePosition": "Late", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": 3, "reverse_linkage_adjustment_components": None, "linkage_W_2D": False}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-04_EXP13_LateLateGCAdj3")
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Test 14
exp_number = 11
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": True, "auxilaryGraph": True,
                 "linkage_adjustment_components": 3, "reverse_linkage_adjustment_components": None, "linkage_W_2D": False}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-04_EXP14_NoneLateEmbGCAdj3")
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Test 15
exp_number = 12
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": True, "auxilaryGraph": True,
                 "linkage_adjustment_components": 3, "reverse_linkage_adjustment_components": None, "linkage_W_2D": False}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-04_EXP15_NoneLateEmbGCAdj3")
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)

