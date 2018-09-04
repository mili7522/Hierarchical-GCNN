import pandas as pd
import os

os.chdir('Data')

### SA1
features = pd.read_csv('2018-09-02-NSW-SA1Input-Normalised.csv', index_col = 0)
SA1s = features.index

links = pd.read_csv("Geography/2018-09-01-NSW-Neighbouring_Suburbs_With_Bridges.csv")
links = links[links['src_SA1_7DIG16'].isin(SA1s) & links['nbr_SA1_7DIG16'].isin(SA1s)]

links.to_csv("Geography/2018-09-01-NSW-Neighbouring_Suburbs_With_Bridges-Filtered.csv", index = None)

### SA2
features = pd.read_csv('2018-08-28-NSW-SA2Input-Normalised.csv', index_col = 0)
SA2s = features.index

links = pd.read_csv("Geography/2018-09-01-SA2Neighbouring_Suburbs_With_Bridges.csv")
links = links[links['src_SA2_MAIN16'].isin(SA2s) & links['nbr_SA2_MAIN16'].isin(SA2s)]

links.to_csv("Geography/2018-09-01-SA2Neighbouring_Suburbs_With_Bridges-Filtered.csv", index = None)


### SA3
features = pd.read_csv('2018-09-05-NSW-SA3Input-Normalised.csv', index_col = 0)
SA3s = features.index

links = pd.read_csv("Geography/SA3_2016_NEIGHBOURS_expanded.csv")
links = links[links['SA3_CODE16'].isin(SA3s) & links['Neighbour'].isin(SA3s)]

links.to_csv("Geography/2018-09-05-SA3Neighbouring_Suburbs-Filtered.csv", index = None)


### SA4
features = pd.read_csv('2018-09-05-NSW-SA4Input-Normalised.csv', index_col = 0)
SA4s = features.index

links = pd.read_csv("Geography/SA4_2016_NEIGHBOURS_expanded.csv")
links = links[links['SA4_CODE16'].isin(SA4s) & links['Neighbour'].isin(SA4s)]

links.to_csv("Geography/2018-09-05-SA4Neighbouring_Suburbs-Filtered.csv", index = None)