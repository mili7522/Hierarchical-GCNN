import pandas as pd
import numpy as np

SA2s = pd.read_csv('Geography/SA2_2016_AUST.csv')
SA2s.set_index('SA2_MAINCODE_2016', inplace = True)

SA2_FEATURES = pd.read_csv('SA2_FEATURES_2018.csv', na_values = ['#VALUE!', '#DIV/0!'])
                                                             
SA2_FEATURES.set_index('SA1_2016', inplace = True)
SA2_FEATURES.dropna(0, 'any', inplace = True)


nswSA2s = SA2_FEATURES.loc[SA2s['GCCSA_NAME_2016'].isin(['Rest of NSW', 'Greater Sydney'])]



nswSA2s['2PP'] = 0

nswSA2s.to_csv('2018-08-28-NSW-SA2Input.csv', index = True)
