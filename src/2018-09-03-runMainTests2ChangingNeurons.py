import ggcnn.experiment as experiment
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import sklearn
from collections import defaultdict
import os


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
                    auxilaryGraph = False, linkage_adjustment_components = None):
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
    
    def create_network(self, net, input):
        net.create_network(input)

        if self.linkage_adjustment_components is not None:
            net.make_linkage_adjustment_layer()
        
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
        
    # # Semi-supervised
    # supervised = False
    # for i in range(no_folds):
    #     results, idx_split = run(no_folds, supervised, i, l, n, expParameters)
    #     predictions_df, r2, corr = getPrediction(idx_split, results)
    #     resultsDict = appendResults(resultsDict, results, r2, corr, i, supervised, no_folds)
    #     saveResults(results, predictions_df, resultsFolder, exp_name, "_Semisupervised-5-{}".format(i))

    # no_folds = 10
    # for i in range(no_folds):
    #     results, idx_split = run(no_folds, supervised, i, l, n, expParameters)
    #     predictions_df, r2, corr = getPrediction(idx_split, results)
    #     resultsDict = appendResults(resultsDict, results, r2, corr, i, supervised, no_folds)
    #     saveResults(results, predictions_df, resultsFolder, exp_name, "_Semisupervised-10-{}".format(i))

    numberOfResults = len(resultsDict['min_loss'])
    otherParams = dict( [(k, [v] * numberOfResults) for k,v in expParameters.items()])
    otherParamsDf = pd.DataFrame(otherParams)
    l_and_n_df = pd.DataFrame({'l': [l] * numberOfResults, 'n': [n] * numberOfResults})
    df = pd.DataFrame(resultsDict)
    df = pd.concat([df, otherParamsDf, l_and_n_df], axis = 1)

    return df


dfs = []
dfSaveName = "Results/2018-09-03-MainTests2ChangingNeurons.csv"

### Test 101
exp_number = 101
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "None", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP101_NoAuxilary_L1N64", l = 1, n = 64)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Test 201
exp_number = 201
expParameters = {"reverseLinkagePosition": "Early", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP201_EarlyLateGC_L1N64", l = 1, n = 64)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Test 501
exp_number = 501
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": False,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP501_NoneLate_L1N64", l = 1, n = 64)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)


### Test 102
exp_number = 102
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "None", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP102_NoAuxilary_L1N128", l = 1, n = 128)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Test 202
exp_number = 202
expParameters = {"reverseLinkagePosition": "Early", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP202_EarlyLateGC_L1N128", l = 1, n = 128)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Test 502
exp_number = 502
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": False,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP502_NoneLate_L1N128", l = 1, n = 128)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)


### Test 103
exp_number = 103
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "None", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP103_NoAuxilary_L1N256", l = 1, n = 256)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Test 203
exp_number = 203
expParameters = {"reverseLinkagePosition": "Early", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP203_EarlyLateGC_L1N256", l = 1, n = 256)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Test 503
exp_number = 503
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": False,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP503_NoneLate_L1N256", l = 1, n = 256)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)


### Test 104
exp_number = 104
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "None", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP104_NoAuxilary_L2N64", l = 2, n = 64)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Test 204
exp_number = 204
expParameters = {"reverseLinkagePosition": "Early", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP204_EarlyLateGC_L2N64", l = 2, n = 64)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Test 504
exp_number = 504
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": False,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP504_NoneLate_L2N64", l = 2, n = 64)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)


### Test 106
exp_number = 106
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "None", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP106_NoAuxilary_L2N256", l = 2, n = 256)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)

### Test 206
exp_number = 206
expParameters = {"reverseLinkagePosition": "Early", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP206_EarlyLateGC_L2N256", l = 2, n = 256)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)

### Test 506
exp_number = 506
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": False,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP506_NoneLate_L2N256", l = 2, n = 256)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)


### Test 107
exp_number = 107
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "None", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP107_NoAuxilary_L3N64", l = 3, n = 64)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)

### Test 207
exp_number = 207
expParameters = {"reverseLinkagePosition": "Early", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP207_EarlyLateGC_L3N64", l = 3, n = 64)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)

### Test 507
exp_number = 507
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": False,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP507_NoneLate_L3N64", l = 3, n = 64)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)


### Test 108
exp_number = 108
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "None", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP108_NoAuxilary_L3N128", l = 3, n = 128)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)

### Test 208
exp_number = 208
expParameters = {"reverseLinkagePosition": "Early", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP208_EarlyLateGC_L3N128", l = 3, n = 128)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)

### Test 508
exp_number = 508
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": False,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP508_NoneLate_L3N128", l = 3, n = 128)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)


### Test 109
exp_number = 109
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "None", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP109_NoAuxilary_L3N256", l = 3, n = 256)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)

### Test 209
exp_number = 209
expParameters = {"reverseLinkagePosition": "Early", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": True,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP209_EarlyLateGC_L3N256", l = 3, n = 256)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)

### Test 509
exp_number = 509
expParameters = {"reverseLinkagePosition": "None", "linkagePosition": "Late", "linkageActFun": False, "linkageBatchNorm": True,
                 "linkageNeurons": None, "auxilaryEmbedding1": False, "auxilaryEmbedding2": False, "auxilaryGraph": False,
                 "linkage_adjustment_components": None}
print(exp_number, expParameters)
df = runBatch(expParameters = expParameters, exp_name = "2018-09-03_EXP509_NoneLate_L3N256", l = 3, n = 256)
dfs.append( pd.concat([pd.DataFrame({"ExpNo": [exp_number]*len(df)}), df], axis = 1) )

### Combine and output
df = pd.concat(dfs); df.to_csv(dfSaveName)