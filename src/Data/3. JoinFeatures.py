import pandas as pd
import numpy as np
import os

os.chdir('Data')

SA1s = pd.read_csv('Geography/SA1_2016_AUST.csv')
SA1s.set_index('SA1_7DIGITCODE_2016', inplace = True)

SA1_FEATURES = pd.read_csv('SA1_FEATURES_2018_NSW.csv', na_values = ['#VALUE!', '#DIV/0!'], index_col = 0)
                                                             
SA1_FEATURES.set_index('SA1_7DIGITCODE_2016', inplace = True)
SA1_FEATURES.dropna(0, 'any', inplace = True)


nswSA1s = SA1_FEATURES.loc[SA1s['GCCSA_NAME_2016'].isin(['Rest of NSW', 'Greater Sydney'])]

nswSA1s['SA1_2PP'] = -1  # Indicating missing value


SA1sAEC = pd.read_csv('SA1_AEC_FINAL_pc.csv', na_values = ['#VALUE!', '#DIV/0!'])
SA1sAEC.set_index('SA1_7DIGITCODE_2016', inplace = True)

nswSA1s_with_no_aec_info = ~nswSA1s.index.isin(SA1sAEC.index)
print(np.sum(nswSA1s_with_no_aec_info))

combinedSA1s = pd.concat([SA1sAEC, nswSA1s.loc[nswSA1s_with_no_aec_info]], axis = 0)


combinedSA1s.to_csv('2018-09-02-NSW-SA1Input.csv', index = True)
