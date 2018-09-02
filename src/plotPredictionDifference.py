# Visualise
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


SA1s = gpd.read_file('../../../Data - Initial Testing/Geography/1270055001_sa1_2016_aust_shape/SA1_2016_AUST.shp')
SA1s = SA1s[SA1s['GCC_NAME16'].isin(['Rest of NSW', 'Greater Sydney'])]
# SA1s = SA1s[SA1s['GCC_NAME16'].isin(['Greater Sydney'])]


results_df = pd.read_csv('Results/PredictionsOneGraphConvolutionAdjSemisupervised.csv', index_col = 0)
test_results = results_df[results_df['idx_split'] == 1]
test_results = test_results[test_results.index.isin(SA1s['SA1_7DIG16'])]

prediction = test_results['Prediction']
results = test_results['Actual']
prediction_diff = prediction - results


#SA1s.set_index('SA1_7DIG16', inplace = True)
SA1s_sub = SA1s[SA1s['SA1_7DIG16'].isin(map(str,test_results.index))].copy()

SA1s_sub['PredictionDiff'] = prediction_diff.values
SA1s_sub['Actual'] = results
SA1s_sub['Prediction'] = prediction


# largest_absolute = max(np.max(prediction_diff), -np.min(prediction_diff))
largest_absolute = 38.01
vmin = -largest_absolute
vmax = largest_absolute
cmp = plt.get_cmap('RdYlGn')  # Diverging
# cmp = plt.get_cmap('viridis')
ax = SA1s_sub.plot(column = 'PredictionDiff', cmap = cmp, vmin = vmin, vmax = vmax)
plt.axis('off')
fig = ax.get_figure()
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
#norm = matplotlib.colors.BoundaryNorm(np.arange(vmin, vmax + 1), cmp.N)   # Discrete colour
#sm = plt.cm.ScalarMappable(cmap = cmp, norm = norm)
sm = plt.cm.ScalarMappable(cmap = cmp, norm = plt.Normalize(vmin, vmax))
sm._A = []
fig.colorbar(sm, cax=cax)
plt.savefig('Results/2018-09-02-SA1OneGraphConvolutionAdjSemisupervised.png', dpi = 300, format = 'png', bbox_inches = 'tight')
plt.close()

# Histogram of differences
#plt.hist(prediction_diff, bins = np.max(prediction_diff) - np.min(prediction_diff), align = 'left')
#plt.savefig('Output/2018-06-08-SA1PredictionDiffHist-SemiSupervisedMelb.png', dpi = 300, format = 'png', bbox_inches = 'tight')