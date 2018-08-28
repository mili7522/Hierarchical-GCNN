import pandas as pd

data = pd.read_csv('2018-06-01-NSW-SA1Input.csv', index_col = 0)

# Filter just those SA1s in the giant component
SA1s = pd.read_csv('Geography/2018-06-01-NSW-SA1s.csv', squeeze = True, header = None)

data = data.loc[SA1s]

mean = data.mean()
std = data.std()

data_normalised = (data - mean) / std
data_normalised['Category'] = data['Category']

data_normalised.to_csv('2018-06-01-NSW-SA1Input-Normalised.csv')


### VIC

data = pd.read_csv('2018-06-07-VIC-SA1Input.csv', index_col = 0)

# Filter just those SA1s in the giant component
SA1s = pd.read_csv('Geography/2018-06-07-VIC-SA1s.csv', squeeze = True, header = None)

data = data.loc[SA1s]

mean = data.mean()
std = data.std()

data_normalised = (data - mean) / std
data_normalised['Category'] = data['Category']

data_normalised.to_csv('2018-06-07-VIC-SA1Input-Normalised.csv')