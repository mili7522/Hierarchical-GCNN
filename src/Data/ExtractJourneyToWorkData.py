import pandas as pd
import numpy as np
import os

os.chdir('Data')

ttw = pd.read_csv("Unused/SA2TTW.csv", skiprows = [1], index_col = 0)  # Residence as columns, work as rows
ttw.index.names = ['SA2 (POW)']

SA2s = pd.read_csv('Geography/SA2_2016_AUST.csv')
SA2s = SA2s[SA2s['GCCSA_NAME_2016'].isin(['Rest of NSW', 'Greater Sydney'])]

ttw = ttw.iloc[ttw.index.isin(SA2s['SA2_NAME_2016']), ttw.columns.isin(SA2s['SA2_NAME_2016'])]

SA2s.set_index('SA2_NAME_2016', inplace = True)

ttw.index = SA2s.loc[ttw.index]['SA2_MAINCODE_2016']
ttw.columns = SA2s.loc[ttw.columns]['SA2_MAINCODE_2016']

ttw = ttw / np.max(np.max(ttw))
ttw.to_csv("SA2TTW.csv")

# Reorder as the same as the features file
# features = pd.read_csv("2018-08-28-NSW-SA2Input-Normalised.csv", index_col = 0)
# features_SA2s = features.index
# assert len(features_SA2s)  == len(ttw.index)