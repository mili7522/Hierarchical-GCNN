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

nswSA3s = SA3_FEATURES.loc[SA3s['GCCSA_NAME_2016'].isin(['Rest of NSW', 'Greater Sydney'])].copy()

nswSA3s['2PP'] = 0

nswSA3s.to_csv('2018-09-05-NSW-SA3Input.csv', index = True)

# Add expanded features
additional_features_df = pd.read_csv('SA3_MBS_Grouped.csv', index_col = 0)
additional_features_df = additional_features_df[additional_features_df.index.isin(nswSA3s.index)]

nswSA3s.drop('2PP', axis = 1, inplace = True)

joined_nsw_df = nswSA3s.join(additional_features_df, how = 'left')  # 3 SA3s with no additional information. Just fill with the average
average = additional_features_df.mean()
joined_nsw_df.fillna(average, inplace = True)

joined_nsw_df['2PP'] = 0
joined_nsw_df.to_csv('2018-09-05-NSW-SA3Input-Expanded.csv', index = True)


### SA4s

SA4s = pd.read_csv('Geography/SA4_2016_AUST.csv')
SA4s.set_index('SA4_CODE_2016', inplace = True)

SA4_FEATURES = pd.read_csv('SA4_FEATURES.csv', na_values = ['#VALUE!', '#DIV/0!'], index_col = 0)
                                                             
SA4_FEATURES.set_index('SA4s', inplace = True)
SA4_FEATURES.index = SA4_FEATURES.index.astype(int)
SA4_FEATURES.dropna(0, 'any', inplace = True)

nswSA4s = SA4_FEATURES.loc[SA4s['GCCSA_NAME_2016'].isin(['Rest of NSW', 'Greater Sydney'])].copy()

nswSA4s['2PP'] = 0

nswSA4s.to_csv('2018-09-05-NSW-SA4Input.csv', index = True)

# Add expanded features
additional_features_df = pd.read_csv('SA4_TAX_DATA_2.csv', index_col = 0)
additional_features_df = additional_features_df[additional_features_df.index.isin(nswSA4s.index)]
additional_features_df.drop('SA4_NAME_2016', axis = 1, inplace = True)
additional_features_df.drop('Unnamed: 2', axis = 1, inplace = True)
additional_features_df.drop('SA4_2011', axis = 1, inplace = True)

nswSA4s.drop('2PP', axis = 1, inplace = True)

joined_nsw_df = nswSA4s.join(additional_features_df, how = 'left')

joined_nsw_df['2PP'] = 0
joined_nsw_df.to_csv('2018-09-05-NSW-SA4Input-Expanded.csv', index = True)