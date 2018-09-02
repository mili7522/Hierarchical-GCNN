import pandas as pd

SA1s = pd.read_csv('2018-09-02-NSW-SA1Input-Normalised.csv', index_col = 0).index

SA2s = pd.read_csv('2018-08-28-NSW-SA2Input-Normalised.csv', index_col = 0).index

linkFile = pd.read_csv('Geography/SA1_2016_AUST.csv', usecols=[1,2])

linkFile = linkFile[linkFile['SA1_7DIGITCODE_2016'].isin(SA1s) & linkFile['SA2_MAINCODE_2016'].isin(SA2s)]

linkFile.to_csv('2018-09-02-SA1SA2Links.csv', index = False)