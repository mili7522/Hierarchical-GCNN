# Visualise
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


SA1s = gpd.read_file('../../../Data - Initial Testing/Geography/1270055001_sa1_2016_aust_shape/SA1_2016_AUST.shp')
SA1s = SA1s[SA1s['GCC_NAME16'].isin(['Rest of NSW', 'Greater Sydney'])]
# SA1s = SA1s[SA1s['GCC_NAME16'].isin(['Greater Sydney'])]

resultFiles1 = ["2018-09-03_EXP1_NoAuxilary_Semisupervised-10-{}".format(i) for i in range(10)]
resultFiles2 = ["2018-09-03_EXP4_LateLateEmbGC_Semisupervised-10-{}".format(i) for i in range(10)]

prediction_diff_comb_1 = []
for resultFile1 in resultFiles1:
    results_df1 = pd.read_csv('Results/Predictions/' + resultFile1 + '.csv', index_col = 0)
    test_results1 = results_df1[results_df1['idx_split'] == 1]  # 1 are the test indexes
    test_results1 = test_results1[test_results1.index.isin(SA1s['SA1_7DIG16'])]
    prediction_diff1 = test_results1['Prediction'] - test_results1['Actual']
    prediction_diff_comb_1.append(prediction_diff1)
    

prediction_diff1 = pd.concat(prediction_diff_comb_1, axis = 1).mean(axis = 1)

prediction_diff_comb_2 = []
for resultFile2 in resultFiles2:
    results_df2 = pd.read_csv('Results/Predictions/' + resultFile2 + '.csv', index_col = 0)
    test_results2 = results_df2[results_df2['idx_split'] == 1]  # 1 are the test indexes
    test_results2 = test_results2[test_results2.index.isin(SA1s['SA1_7DIG16'])]
    prediction_diff2 = test_results2['Prediction'] - test_results2['Actual']
    prediction_diff_comb_2.append(prediction_diff2)

prediction_diff2 = pd.concat(prediction_diff_comb_2, axis = 1).mean(axis = 1)


#SA1s.set_index('SA1_7DIG16', inplace = True)
SA1s_sub = SA1s[SA1s['SA1_7DIG16'].isin(map(str,prediction_diff1.index))].copy()

SA1s_sub['PredictionDiffDiff'] = np.abs(prediction_diff1.values) - np.abs(prediction_diff2.values)

# Clip results
SA1s_sub['PredictionDiffDiff'].clip_upper(5, inplace = True)
SA1s_sub['PredictionDiffDiff'].clip_lower(-5, inplace = True)
#

largest_absolute = max(SA1s_sub['PredictionDiffDiff'].max(), -SA1s_sub['PredictionDiffDiff'].min())
vmin = -largest_absolute
vmax = largest_absolute
cmp = plt.get_cmap('RdYlGn')  # Diverging
# cmp = plt.get_cmap('viridis')
ax = SA1s_sub.plot(column = 'PredictionDiffDiff', cmap = cmp, vmin = vmin, vmax = vmax)
plt.axis('off')
# ax.set_title('Difference between prediction error\nfrom hierarchical graph-cnn vs standard graph-cnn')
fig = ax.get_figure()
# cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.9, 0.07, 0.05, 0.7])
fig.subplots_adjust(hspace=0.0, wspace=0.0)
#norm = matplotlib.colors.BoundaryNorm(np.arange(vmin, vmax + 1), cmp.N)   # Discrete colour
#sm = plt.cm.ScalarMappable(cmap = cmp, norm = norm)
sm = plt.cm.ScalarMappable(cmap = cmp, norm = plt.Normalize(vmin, vmax))
sm._A = []
fig.colorbar(sm, cax=cax)
plt.savefig('Results/Visualisations/' + resultFile1 + '-' + resultFile2 + '-AVG.png', dpi = 300, format = 'png', bbox_inches = 'tight')
plt.close()

# Histogram of differences
#plt.hist(prediction_diff, bins = np.max(prediction_diff) - np.min(prediction_diff), align = 'left')
#plt.savefig('Output/2018-06-08-SA1PredictionDiffHist-SemiSupervisedMelb.png', dpi = 300, format = 'png', bbox_inches = 'tight')