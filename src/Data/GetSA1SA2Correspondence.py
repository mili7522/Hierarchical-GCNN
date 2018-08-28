import pandas as pd

SA1Links = pd.read_csv('2018-08-25-NSW-NeighbourDistance.csv')
SA1s = pd.unique(SA1Links[['src_SA1_7DIG16', 'nbr_SA1_7DIG16']].values.ravel())

SA2Links = pd.read_csv('Geography/2018-08-28-NSW-SA2_Neighbouring_Suburbs_With_Bridges-GCC.csv')
SA2s = pd.unique(SA2Links[['src_SA2_MAIN16', 'nbr_SA2_MAIN16']].values.ravel())

linkFile = pd.read_csv('Geography/SA1_2016_AUST.csv', usecols=[1,2])

linkFile = linkFile[linkFile['SA1_7DIGITCODE_2016'].isin(SA1s) & linkFile['SA2_MAINCODE_2016'].isin(SA2s)]

linkFile.to_csv('SA1SA2Links.csv', index = False)