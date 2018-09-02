import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import geopandas as gpd
import os

os.chdir('Data')

nsw = gpd.read_file('../../../../Data - Initial Testing/Geography/1270055001_sa1_2016_aust_shape/SA1_2016_AUST.shp')
nsw = nsw[nsw['GCC_NAME16'].isin(['Rest of NSW', 'Greater Sydney'])]

features = pd.read_csv('2018-09-02-NSW-SA1Input-Normalised.csv', index_col = 0)
features = features[features.index.isin(nsw['SA1_7DIG16'])]
linkFile = "Geography/2018-09-01-NSW-Neighbouring_Suburbs_With_Bridges-Filtered.csv"
centresFile = "Geography/2018-09-01-NSW-SA1-2016Centres.csv"

ax = plt.axes([0, 0, 1, 1],
              projection=ccrs.Mercator())

# ax.set_extent([149.85, 151.7, -34.4, -33], ccrs.Geodetic())  # Sydney
ax.set_extent([141.0, 153.7, -37.4, -28.0], ccrs.Geodetic())  # NSW

cmp = plt.get_cmap('Greens')  # Colour map. Also can use 'jet', 'brg', 'rainbow', 'winter', etc
# colours = cmp(np.linspace(0,1.0, 10))

max_value = features['SA1_2PP'].max()
for sa1, geometry in zip(nsw.SA1_7DIG16, nsw.geometry):
    try:
        value = features.loc[int(sa1)].loc['SA1_2PP']
        if value == -1:
            facecolor = 'white'
        else:
            value = value / max_value
            facecolor = cmp(value)
    except:
        facecolor = 'white'
    edgecolor = 'white'

    ax.add_geometries([geometry], ccrs.PlateCarree(),
                      facecolor=facecolor, edgecolor=edgecolor, linewidth = 0.1)


linkData = pd.read_csv(linkFile)
centresData = pd.read_csv(centresFile)

lats = centresData.set_index("SA1_7DIG16").loc[:,"0"]
longs = centresData.set_index("SA1_7DIG16").loc[:,"1"]


for edge in linkData.itertuples():
    start = edge[1]
    end = edge[2]
    # if (not start in features.index) or (not end in features.index):
        # continue
    try:
        lat = [lats[start], lats[end]]
        long = [longs[start], longs[end]]
    except:
        continue
    track = sgeom.LineString(zip(long, lat))
    ax.add_geometries([track], ccrs.PlateCarree(),
                      facecolor='none', edgecolor = 'black', linewidth = 0.1)


sm = plt.cm.ScalarMappable(cmap=cmp,norm=plt.Normalize(0,max_value))
sm._A = []
plt.colorbar(sm,ax=ax, fraction = 0.03, label = 'Labor Percentage - Two Party Preferred')

# ax.set_title('Distribution of Households over Suburbs')

plt.gca().outline_patch.set_visible(False)
plt.savefig('NSWSA1WithLinks.png', dpi = 300, format = 'png', bbox_inches = 'tight')