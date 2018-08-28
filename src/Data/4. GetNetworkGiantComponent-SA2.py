import networkx as nx
import pandas as pd
import os

#os.chdir('Data')


def getGCC(SA2s, links, linksSaveName, SA2saveName):
    # Filter to only included SA2s
    links = links[links['src_SA2_MAIN16'].isin(SA2s) & links['nbr_SA2_MAIN16'].isin(SA2s)]
    
    G = nx.Graph()
    
    for edge in links.itertuples():
        if edge[3] > 1E-5:  # Filter out SA2s just touching by the corners
            G.add_edge(edge[1], edge[2])
    
    Gcc = sorted(nx.connected_component_subgraphs(G), key = len, reverse = True)
    G0 = Gcc[0]
    
    
    # Save links of giant component
    GccSA2s = pd.Series(list(G0.nodes))
    
    links = links[links['src_SA2_MAIN16'].isin(GccSA2s) & links['nbr_SA2_MAIN16'].isin(GccSA2s)]
    links = links[links['LENGTH'] > 1E-5]  # Filter out SA2s just touching by the corners
    links.to_csv(linksSaveName, index = None)
    
    # Save SA2s of giant component
    SA2s = SA2s[SA2s.isin(GccSA2s)]
    SA2s.to_csv(SA2saveName, index = None)


### NSW
linkFile = "Geography/2018-08-28-NSW-SA2_Neighbouring_Suburbs_With_Bridges.csv"
links = pd.read_csv(linkFile)
SA2s = pd.read_csv('2018-08-28-NSW-SA2Input.csv', usecols = [0], squeeze = True)
getGCC(SA2s, links, 'Geography/2018-08-28-NSW-SA2_Neighbouring_Suburbs_With_Bridges-GCC.csv', 'Geography/2018-08-28-NSW-SA2s.csv')