import pandas as pd
import numpy as np

df = pd.read_csv('Data/NewData/SA1-HouseholdMobility.csv', index_col=0)

total = df['All residents in the household aged one year and over had a different address one year ago'] + \
        df['Some residents in the household aged one year and over had a different address one year ago'] + \
        df['No residents in the household aged one year and over had a different address one year ago']

df['Predict'] = df['All residents in the household aged one year and over had a different address one year ago'] / total * 100
    
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

data_normalised['Predict'] = df['Predict']
data_normalised['Predict'].fillna(-1, inplace = True)  # To indicate missing value


data_normalised.to_csv('Data/NewData/2018-11-13-NSW-SA1Input-HousingMobility-Normalised.csv', index = True)
