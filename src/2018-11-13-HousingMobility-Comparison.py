import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import sklearn
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.utils import to_categorical
import sys
import time
import numpy as np

### Import data

#ls = [1, 2, 3, 4, 5, 6]
#ns = [8, 16, 32, 64, 128]
#i = int(sys.argv[1])
#l = ls[i // len(ns)]
#n = ns[i % len(ns)]

l = 2
n = 128
i = 0


saveName = 'Results/2018-11-13-DeepComparison-HousingMobility-l={:d}-n={:d}.csv'.format(l,n)

###
data = pd.read_csv('Data/NewData/2018-11-13-NSW-SA1Input-HousingMobility-Normalised.csv')

data = data[data.iloc[:,-1] >= 0]  # Exclude invalide values

#prediction = np.floor(data.iloc[:,-1].values / 10).astype(int)
prediction = data.iloc[:,-1].values
training_data = data.iloc[:,1:-1].values  # Exclude SA1_MAINCODE_2016 and Category

no_folds = 5
supervised = True
inst = KFold(n_splits = no_folds, shuffle=True, random_state=125)

loss = []
val_loss = []
min_val_loss = []
acc = []
val_acc = []
neurons = []
layers = []
rep = []
supervised_vals = []
no_folds_vals = []
r2 = []
corr = []

for i in range(no_folds):
    startTime = time.time()
    
    if supervised:
        train_idx, test_idx = list(inst.split(np.arange(len(prediction))))[i]
    else:
        test_idx, train_idx = list(inst.split(np.arange(len(prediction))))[i]
        
    train_x = training_data[train_idx]
    train_y = prediction[train_idx]
    val_x = training_data[test_idx]
    val_y = prediction[test_idx]
    
    ### Build model
    K.clear_session()
    
    def buildModel(neurons, layers = 1):
        inputs = Input(shape = (train_x.shape[-1],))
        
        x = Dense(neurons, use_bias = True)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        for j in range(layers):
            x = Dense(neurons, use_bias = True)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)

            x = Dense(neurons, use_bias = True)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
            
        x = Dense(neurons, use_bias = True)(x)
        x = BatchNormalization()(x)
            
#        predictions = Dense(10, activation = 'softmax')(x)
        predictions = Dense(1, activation = 'linear')(x)
        
        model = Model(inputs = inputs, outputs = predictions)
    
        model.compile(optimizer='adam',
#                      loss='categorical_crossentropy',
#                      metrics=['accuracy'])
                      loss='mean_squared_error')
        
        return model
    
    model = buildModel(neurons = n, layers = l)
    
    history = model.fit(train_x,
#                        to_categorical(train_y, 10),
                        train_y,
                        epochs=1000,
                        batch_size=64,
#                        validation_data = (val_x, to_categorical(val_y, 10)))
                        validation_data = (val_x, val_y))
    
    predictions = model.predict(val_x).ravel()


    loss.append(history.history['loss'][-1])
    val_loss.append(history.history['val_loss'][-1])
    min_val_loss.append(np.min(history.history['val_loss']))
    # acc.append(history.history['acc'][-1])
    # val_acc.append(history.history['val_acc'][-1])
    rep.append(i)
    layers.append(l)
    neurons.append(n)
    supervised_vals.append(supervised)
    no_folds_vals.append(no_folds)
    r2.append(sklearn.metrics.r2_score(predictions, val_y.ravel()))
    corr.append(np.corrcoef(predictions, val_y.ravel())[1,0])
    
    endTime = time.time()
    print('Time:', (endTime - startTime)/60)
    
    
loss = pd.DataFrame(loss, columns = ['Loss'])
val_loss = pd.DataFrame(val_loss, columns = ['Validation Loss'])
min_val_loss = pd.DataFrame(min_val_loss, columns = ['Min Validation Loss'])
# acc = pd.DataFrame(acc, columns = ['Accuracy'])
# val_acc = pd.DataFrame(val_acc, columns = ['Validation Accuracy'])
rep = pd.DataFrame(rep, columns = ['Repeat'])
l = pd.DataFrame(layers, columns = ['Layers'])
n = pd.DataFrame(neurons, columns = ['Neurons'])
no_folds = pd.DataFrame(no_folds_vals, columns = ['No Folds'])
supervised = pd.DataFrame(supervised_vals, columns = ['Supervised'])
r2 = pd.DataFrame(r2, columns = ['R2'])
corr = pd.DataFrame(corr, columns = ['Correlation'])

df = pd.concat([n, l, no_folds, supervised, rep, min_val_loss, val_loss, loss, r2, corr], axis = 1)

print(df)