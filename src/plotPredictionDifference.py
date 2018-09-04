# Visualise
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import geopandas as gpd
import os


SA1s = gpd.read_file('../../../Data - Initial Testing/Geography/1270055001_sa1_2016_aust_shape/SA1_2016_AUST.shp')
SA1s = SA1s[SA1s['GCC_NAME16'].isin(['Rest of NSW', 'Greater Sydney'])]
# SA1s = SA1s[SA1s['GCC_NAME16'].isin(['Greater Sydney'])]
# SA1s = SA1s[SA1s['GCC_NAME16'].isin(['Greater Sydney'])]

# resultFile = "2018-09-03_EXP1_NoAuxilary_Semisupervised-10-0"
# resultFile = "2018-09-03_EXP4_LateLateEmbGC_Semisupervised-10-0"

# results_df = pd.read_csv('Results/Predictions/' + resultFile + '.csv', index_col = 0)
# test_results = results_df[results_df['idx_split'] == 1]  # 1 are the test indexes
# test_results = test_results[test_results.index.isin(SA1s['SA1_7DIG16'])]

# prediction = test_results['Prediction']
# results = test_results['Actual']
# prediction_diff = prediction - results

### Averaging over several files:
resultFiles = ["2018-09-03_EXP4_LateLateEmbGC_Semisupervised-10-{}".format(i) for i in range(10)]
prediction_diff_comb = []
prediction_comb = []
for resultFile in resultFiles:
    results_df = pd.read_csv('Results/Predictions/' + resultFile + '.csv', index_col = 0)
    test_results = results_df[results_df['idx_split'] == 1]  # 1 are the test indexes
    test_results = test_results[test_results.index.isin(SA1s['SA1_7DIG16'])]
    prediction_diff = test_results['Prediction'] - test_results['Actual']
    prediction_diff_comb.append(prediction_diff)
    prediction_comb.append(test_results['Prediction'])
prediction_diff = pd.concat(prediction_diff_comb, axis = 1).mean(axis = 1)
prediction = pd.concat(prediction_comb, axis = 1).mean(axis = 1)


#SA1s.set_index('SA1_7DIG16', inplace = True)
SA1s_sub = SA1s[SA1s['SA1_7DIG16'].isin(map(str,prediction.index))].copy()

# SA1s_sub['PredictionDiff'] = np.abs(prediction_diff.values)
# SA1s_sub['Actual'] = results

SA1s_sub['Prediction'] = prediction.values

# Clip results
# SA1s_sub['PredictionDiff'].clip_upper(20, inplace = True)
#

#####
# largest_absolute = max(SA1s_sub['PredictionDiff'].max(), -SA1s_sub['PredictionDiff'].min())
# vmin = 0
# vmax = largest_absolute
# # cmp = plt.get_cmap('RdYlGn')  # Diverging
# cmp = plt.get_cmap('viridis')
cmp = plt.get_cmap('Reds')
# ax = SA1s_sub.plot(column = 'PredictionDiff', cmap = cmp, vmin = vmin, vmax = vmax)
# plt.axis('off')
# fig = ax.get_figure()
# cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
# #norm = matplotlib.colors.BoundaryNorm(np.arange(vmin, vmax + 1), cmp.N)   # Discrete colour
# #sm = plt.cm.ScalarMappable(cmap = cmp, norm = norm)
# sm = plt.cm.ScalarMappable(cmap = cmp, norm = plt.Normalize(vmin, vmax))
# sm._A = []
# fig.colorbar(sm, cax=cax)
# plt.savefig('Results/Visualisations/' + resultFile + '.png', dpi = 300, format = 'png', bbox_inches = 'tight')
# plt.close()
#####

# Histogram of differences
#plt.hist(prediction_diff, bins = np.max(prediction_diff) - np.min(prediction_diff), align = 'left')
#plt.savefig('Output/2018-06-08-SA1PredictionDiffHist-SemiSupervisedMelb.png', dpi = 300, format = 'png', bbox_inches = 'tight')


#####
ax = plt.axes([0, 0, 1, 1],
              projection=ccrs.Mercator())

# ax.set_extent([149.85, 151.7, -34.4, -33], ccrs.Geodetic())  # Sydney
ax.set_extent([141.0, 153.7, -37.4, -28.0], ccrs.Geodetic())  # NSW

# max_value = SA1s_sub['PredictionDiff'].max()
# min_value = SA1s_sub['PredictionDiff'].min()
max_value = SA1s_sub['Prediction'].max()
min_value = SA1s_sub['Prediction'].min()

for geometry, value in zip(SA1s_sub.geometry, SA1s_sub['Prediction']):
    try:
        value = (value - min_value) / (max_value - min_value)
        facecolor = cmp(value)
    except:
        facecolor = 'white'
    edgecolor = 'white'

    ax.add_geometries([geometry], ccrs.PlateCarree(),
                      facecolor=facecolor, edgecolor=edgecolor, linewidth = 0.1)
###

sm = plt.cm.ScalarMappable(cmap = cmp, norm = plt.Normalize(min_value, max_value))
sm._A = []
# fig.colorbar(sm, cax=cax)
# plt.tight_layout()
plt.colorbar(sm,ax = ax, fraction = 0.03)
plt.gca().outline_patch.set_visible(False)
plt.savefig('Results/Visualisations/' + resultFile + '-AVGPredictionOnly.png', dpi = 300, format = 'png', bbox_inches = 'tight')
plt.close()