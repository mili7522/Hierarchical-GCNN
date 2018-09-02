import geopandas as gpd
import pandas as pd
import os

os.chdir('Data')

austSA1 = gdp.read_file('../../../../Data - Initial Testing/Geography/1270055001_sa1_2016_aust_shape/SA1_2016_AUST.shp')

## Don't change the index directly. This seems to break some of the geopandas functionality
nswSA1 = austSA1[austSA1['GCC_NAME16'].isin(['Rest of NSW', 'Greater Sydney'])]  # Exclude some cases of 'Migratory - Offshore - Shipping (NSW)'

# Save centroids
centroids_x = nswSA1.centroid.x
centroids_y = nswSA1.centroid.y

centroids = pd.DataFrame([centroids_y, centroids_x])
centroids = centroids.T


### Save filtered shapefile
# nswSA1.to_file('Geography/2018-09-01-NSW2016SA1')

SA1_code = nswSA1['SA1_7DIG16']
# SA2_code.to_csv('2018-09-01-NSW2016SA2.csv', index = False)

centroids.index = SA1_code
centroids.to_csv('Geography/2018-09-01-NSW-SA1-2016Centres.csv', index = True)
