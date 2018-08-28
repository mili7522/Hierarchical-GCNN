import pandas as pd
import math


def distanceBetweenCm(coord1, coord2):
    # https://stackoverflow.com/questions/44910530/how-to-find-the-distance-between-2-points-in-2-different-dataframes-in-pandas
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    dLat = math.radians(lat2-lat1)
    dLon = math.radians(lon2-lon1)

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    a = math.sin(dLat/2) * math.sin(dLat/2) + math.sin(dLon/2) * math.sin(dLon/2) * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return c * 6371


links = pd.read_csv('Geography/2018-08-24-NSW-Neighbouring_Suburbs_With_Bridges-GCC.csv')
centres = pd.read_csv('Geography/2018-06-01-NSW-SA1-2016Centres.csv', index_col = 0)


distances = []
for edge in links.itertuples():
    distances.append(distanceBetweenCm(centres.loc[edge[1]], centres.loc[edge[2]]))

links['Distance'] = distances

links.to_csv('2018-08-25-NSW-NeighbourDistance.csv', index = None)