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

nswSA1s['SA1_2PP'] = -1  # To indicate missing AEC value


##### For use with the new AEC file --- Unused
# AEC = pd.read_csv('ALP_2PP_SA12016_FINAL.csv', index_col = 0)

###
# AEC_not_Features = AEC.index[~AEC.index.isin(nswSA1s.index)]
# Features_not_AEC = nswSA1s.index[~nswSA1s.index.isin(AEC.index)]

# AEC_not_Features = AEC_not_Features.sort_values()
# Features_not_AEC = Features_not_AEC.sort_values()

# pd.DataFrame([AEC_not_Features, Features_not_AEC]).transpose().to_csv('MisalignedAECSA1.csv')

# print('Valid SA1s in AEC_not_Features:', np.sum(AEC_not_Features.isin(SA1s.index)), 'of', len(AEC_not_Features))
# print('Features_not_AEC:', len(Features_not_AEC))
###

# AEC = AEC[AEC.index.isin(nswSA1s.index)]

# nswSA1s.loc[AEC.index, 'SA1_2PP'] = AEC['SA1_2PP']

# nswSA1s.to_csv('2018-09-02-NSW-SA1Input.csv', index = True)
#####

aecSA1s = pd.read_csv('SA1_AEC_FINAL_pc.csv', index_col = 0)

sa1sWithoutAEC = nswSA1s[~nswSA1s.index.isin(aecSA1s.index)]

combinedSA1s = pd.concat([aecSA1s, sa1sWithoutAEC], axis = 0)
combinedSA1s.to_csv('2018-09-02-NSW-SA1Input.csv', index = True)
