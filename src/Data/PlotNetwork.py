import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import geopandas as gdp
import os

os.chdir('Data')

nsw = gdp.read_file('Geography/2018-09-01-NSW2016SA2/2018-09-01-NSW2016SA2.shp')

features = pd.read_csv('SA2_FEATURES_2018.csv', index_col = 0)
features = features[features.index.isin(nsw['SA2_MAIN16'])]
linkFile = "Geography/2018-09-01-SA2Neighbouring_Suburbs_With_Bridges.csv"
centresFile = "Geography/2018-09-01-NSW-SA2-2016Centres.csv"

ax = plt.axes([0, 0, 1, 1],
              projection=ccrs.Mercator())

# ax.set_extent([149.85, 151.7, -34.4, -33], ccrs.Geodetic())  # Sydney
ax.set_extent([141.0, 153.7, -37.4, -28.0], ccrs.Geodetic())  # NSW

cmp = plt.get_cmap('Greens')  # Colour map. Also can use 'jet', 'brg', 'rainbow', 'winter', etc
# colours = cmp(np.linspace(0,1.0, 10))

max_value = features['BLUECOLLAR'].max()
for sa2, geometry in zip(nsw.SA2_MAIN16, nsw.geometry):
    try:
        value = features.loc[int(sa2)].loc['BLUECOLLAR'] / max_value
        facecolor = cmp(value)
        
    except:
        facecolor = 'white'
    edgecolor = 'white'

    ax.add_geometries([geometry], ccrs.PlateCarree(),
                      facecolor=facecolor, edgecolor=edgecolor, linewidth = 0.2)


linkData = pd.read_csv(linkFile)
centresData = pd.read_csv(centresFile)

lats = centresData.set_index("SA2_MAIN16").loc[:,"0"]
longs = centresData.set_index("SA2_MAIN16").loc[:,"1"]


for edge in linkData.itertuples():
    start = edge[1]
    end = edge[2]
    lat = [lats[start], lats[end]]
    long = [longs[start], longs[end]]
    track = sgeom.LineString(zip(long, lat))
    ax.add_geometries([track], ccrs.PlateCarree(),
                      facecolor='none', edgecolor = 'black', linewidth = 0.2)


sm = plt.cm.ScalarMappable(cmap=cmp,norm=plt.Normalize(0,max_value))
sm._A = []
plt.colorbar(sm,ax=ax, fraction = 0.03, label = 'Fraction Blue Collar Workers')

# ax.set_title('Distribution of Households over Suburbs')

plt.gca().outline_patch.set_visible(False)
plt.savefig('NSWWithLinks.png', dpi = 300, format = 'png', bbox_inches = 'tight')