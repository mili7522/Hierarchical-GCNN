import pandas as pd
import os

os.chdir('Data')

data = pd.read_csv('2018-08-28-NSW-SA2Input.csv', index_col = 0)

# Filter just those SA1s in the giant component
# SA1s = pd.read_csv('Geography/2018-08-28-NSW-SA2s.csv', squeeze = True, header = None)
# data = data.loc[SA1s]

mean = data.mean()
std = data.std()

data_normalised = (data - mean) / std
data_normalised['2PP'] = data['2PP']

data_normalised.to_csv('2018-08-28-NSW-SA2Input-Normalised.csv')

