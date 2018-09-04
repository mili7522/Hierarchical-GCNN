import pandas as pd
import numpy as np
import os

os.chdir('Data')

SA3s = pd.read_csv('Geography/SA3_2016_AUST.csv')
SA3s.set_index('SA3_CODE_2016', inplace = True)

SA3_FEATURES = pd.read_csv('SA3_FEATURES.csv', na_values = ['#VALUE!', '#DIV/0!'], index_col = 0)
                                                             
SA3_FEATURES.set_index('SA3s', inplace = True)
SA3_FEATURES.index = SA3_FEATURES.index.astype(int)
SA3_FEATURES.dropna(0, 'any', inplace = True)


nswSA3s = SA3_FEATURES.loc[SA3s['GCCSA_NAME_2016'].isin(['Rest of NSW', 'Greater Sydney'])]

nswSA3s['2PP'] = 0

nswSA3s.to_csv('2018-09-05-NSW-SA3Input.csv', index = True)


### SA4s

SA4s = pd.read_csv('Geography/SA4_2016_AUST.csv')
SA4s.set_index('SA4_CODE_2016', inplace = True)

SA4_FEATURES = pd.read_csv('SA4_FEATURES.csv', na_values = ['#VALUE!', '#DIV/0!'], index_col = 0)
                                                             
SA4_FEATURES.set_index('SA4s', inplace = True)
SA4_FEATURES.index = SA4_FEATURES.index.astype(int)
SA4_FEATURES.dropna(0, 'any', inplace = True)


nswSA4s = SA4_FEATURES.loc[SA4s['GCCSA_NAME_2016'].isin(['Rest of NSW', 'Greater Sydney'])]


nswSA4s['2PP'] = 0

nswSA4s.to_csv('2018-09-05-NSW-SA4Input.csv', index = True)