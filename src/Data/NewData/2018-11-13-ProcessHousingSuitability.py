import pandas as pd
import numpy as np

df = pd.read_csv('Data/NewData/SA1-HousingSuitability.csv', index_col=0)

total_households = df['Four or more extra bedrooms needed'] + df['Three extra bedrooms needed'] + \
                   df['Two extra bedrooms needed'] + df['One extra bedroom needed'] + df['No bedrooms needed or spare'] + \
                   df['One bedroom spare'] + df['Two bedrooms spare'] + df['Three bedrooms spare'] + df['Four or more bedrooms spare']

df['Average'] = (df['Four or more bedrooms spare'] * 4 + df['Three bedrooms spare'] * 3 + df['Two bedrooms spare'] * 2 + \
                 df['One bedroom spare'] - df['One extra bedroom needed'] - df['Two extra bedrooms needed'] * 2 - \
                 df['Three extra bedrooms needed'] * 3 - df['Four or more extra bedrooms needed'] * 4) / \
                total_households
    
df.dropna(inplace = True)

###

SA1s = pd.read_csv('Data/Geography/SA1_2016_AUST.csv')
SA1s.set_index('SA1_7DIGITCODE_2016', inplace = True)

SA1_FEATURES = pd.read_csv('Data/SA1_FEATURES_2018_NSW.csv', na_values = ['#VALUE!', '#DIV/0!'], index_col = 0)
                                                             
SA1_FEATURES.set_index('SA1_7DIGITCODE_2016', inplace = True)
SA1_FEATURES.dropna(0, 'any', inplace = True)


nswSA1s = SA1_FEATURES.loc[SA1s['GCCSA_NAME_2016'].isin(['Rest of NSW', 'Greater Sydney'])]

### Normalise

mean = nswSA1s.mean()
std = nswSA1s.std()

data_normalised = (nswSA1s - mean) / std


###

data_normalised['Predict'] = df['Average']
data_normalised['Predict'].fillna(-10, inplace = True)  # To indicate missing value


data_normalised.to_csv('Data/NewData/2018-11-13-NSW-SA1Input-HousingSuitability-Normalised.csv', index = True)
