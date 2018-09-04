import pandas as pd

SA2s = pd.read_csv('2018-08-28-NSW-SA2Input-Normalised.csv', index_col = 0).index

SA3s = pd.read_csv('2018-09-05-NSW-SA3Input-Normalised.csv', index_col = 0).index

SA4s = pd.read_csv('2018-09-05-NSW-SA4Input-Normalised.csv', index_col = 0).index


linkFile = pd.read_csv('Geography/SA2_2016_AUST.csv', usecols=[0,3])

linkFile = linkFile[linkFile['SA2_MAINCODE_2016'].isin(SA2s) & linkFile['SA3_CODE_2016'].isin(SA3s)]

linkFile.to_csv('2018-09-05-SA2SA3Links.csv', index = False)

###
linkFile = pd.read_csv('Geography/SA3_2016_AUST.csv', usecols=[0,2])

linkFile = linkFile[linkFile['SA4_CODE_2016'].isin(SA4s) & linkFile['SA3_CODE_2016'].isin(SA3s)]

linkFile.to_csv('2018-09-05-SA3SA4Links.csv', index = False)