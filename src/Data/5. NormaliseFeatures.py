import pandas as pd
import os

os.chdir('Data')

# data = pd.read_csv('SA1_AEC_FINAL_pc.csv', index_col = 0)
data = pd.read_csv('2018-09-02-NSW-SA1Input.csv', index_col = 0)

# Filter just those SA1s in the giant component
# SA1s = pd.read_csv('Geography/2018-08-24-NSW-SA1s.csv', squeeze = True, header = None)
# data = data.loc[SA1s]

mean = data.mean()
std = data.std()

data_normalised = (data - mean) / std
data_normalised.iloc[:,-1] = data.iloc[:,-1]

data_normalised.to_csv('2018-09-02-NSW-SA1Input-Normalised.csv')

