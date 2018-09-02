import geopandas as gpd
import pandas as pd
import os

os.chdir('Data')

austSA2 = gpd.read_file('Geography/1270055001_sa2_2016_aust_shape/SA2_2016_AUST.shp')


## Don't change the index directly. This seems to break some of the geopandas functionality
nswSA2 = austSA2[austSA2['GCC_NAME16'].isin(['Rest of NSW', 'Greater Sydney'])]  # Exclude some cases of 'Migratory - Offshore - Shipping (NSW)'

# Save centroids
centroids_x = nswSA2.centroid.x
centroids_y = nswSA2.centroid.y

centroids = pd.DataFrame([centroids_y, centroids_x])
centroids = centroids.T


### Save filtered shapefile
nswSA2.to_file('Geography/2018-09-01-NSW2016SA2')

SA2_code = nswSA2['SA2_MAIN16']
# SA2_code.to_csv('2018-09-01-NSW2016SA2.csv', index = False)

centroids.index = SA2_code
centroids.to_csv('Geography/2018-09-01-NSW-SA2-2016Centres.csv', index = True)
