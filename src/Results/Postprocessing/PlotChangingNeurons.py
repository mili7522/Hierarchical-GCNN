import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir('Results')

results = pd.read_csv('2018-09-03-MainTests2ChangingNeurons.csv', index_col = 0)
results['ExpType'] = results['ExpNo']//100

results2 = pd.read_csv('2018-09-03-MainTests1.csv', index_col = 0)
exp1 = results2[(results2['ExpNo'] == 1) & (results2['supervised'] == True) & (results2['no_fold_vals'] == 5)].copy()
exp2 = results2[(results2['ExpNo'] == 2) & (results2['supervised'] == True) & (results2['no_fold_vals'] == 5)].copy()
exp5 = results2[(results2['ExpNo'] == 5) & (results2['supervised'] == True) & (results2['no_fold_vals'] == 5)].copy()
exp1['ExpType'] = 1
exp2['ExpType'] = 2
exp5['ExpType'] = 5

results = pd.concat([results, exp1, exp2, exp5])
results = results[['ExpType', 'l', 'n', 'min_loss']]
results = np.sqrt(results)

# gp = results.groupby(['ExpType', 'l', 'n'])

# mean = gp.mean()
# std = gp.std()

# for t in [1, 2, 5]:
#     for l in [1, 2, 3]:
#         plt.plot(mean.loc[t].loc[l])
# plt.show()

pivot = pd.pivot_table(results, index = ['l', 'n'], values = ['min_loss'], columns = ['ExpType'], aggfunc = [np.mean, np.std])
ax = plt.axes()
pivot['mean'].plot.bar(yerr = pivot['std'], ax = ax, error_kw = {'elinewidth': 0.5, 'capthick': 0.5, 'ecolor': 'black', 'capsize': 1})
plt.legend(['Graph-CNN', 'Hierachical Graph-CNN V1', 'Hierachical Graph-CNN V2'])
plt.show()