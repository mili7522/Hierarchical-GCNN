import pandas as pd
import numpy as np
import os

os.chdir('Data')


SA2s = pd.read_csv('Geography/SA2_2016_AUST.csv')
SA2s.set_index('SA2_MAINCODE_2016', inplace = True)

SA2_FEATURES = pd.read_csv('SA2_FEATURES_2018.csv', na_values = ['#VALUE!', '#DIV/0!'])
                                                             
SA2_FEATURES.set_index('SA1_2016', inplace = True)
SA2_FEATURES.dropna(0, 'any', inplace = True)


nswSA2s = SA2_FEATURES.loc[SA2s['GCCSA_NAME_2016'].isin(['Rest of NSW', 'Greater Sydney'])].copy()



nswSA2s['2PP'] = 0

nswSA2s.to_csv('2018-08-28-NSW-SA2Input.csv', index = True)


### Adding expanded features

additional_features_df = pd.read_csv('SA2_JOURNEY_TO_WORK.csv', index_col = 0)
additional_features_df = additional_features_df.loc[SA2s['GCCSA_NAME_2016'].isin(['Rest of NSW', 'Greater Sydney'])]
additional_features_df.drop('SA2_NAME', axis = 1, inplace = True)

nswSA2s.drop('2PP', axis = 1, inplace = True)

joined_nsw_df = nswSA2s.join(additional_features_df, how = 'inner')
joined_nsw_df.dropna(how = 'any', inplace = True)

joined_nsw_df['2PP'] = 0
joined_nsw_df.to_csv('2018-09-05-NSW-SA2Input-Expanded.csv', index = True)