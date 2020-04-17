from pyts.approximation import SymbolicAggregateApproximation
from scipy.io import loadmat
import numpy as np
import json
import hdf5storage
import matplotlib.pyplot as plt
import scipy.signal as ss



mat = hdf5storage.loadmat('/home/bayrakrg/neurdy/d3/conn/processed_yeo_id108828.mat')
conn = mat['Vp_clean'][0, 0]  #default is the 400 parcellation
del mat

# row-wise subtraction
# requires some stupid transpose operation
# plt.plot(np.transpose(conn_norm))
conn_norm = np.transpose((np.transpose(conn) - np.mean(conn, axis=1)) / np.std(conn, axis=1))
m = 20
conn_norm_ds = np.transpose(ss.resample(np.transpose(conn_norm), m))

transformer = SymbolicAggregateApproximation(n_bins=8, strategy='uniform')
# https://pyts.readthedocs.io/en/stable/modules/approximation.html
# strategy='uniform': all bins in each sample have identical widths,
# strategy='quantile': all bins in each sample have the same number of points,
# strategy='normal': bin edges are quantiles from a standard normal distribution.
data = {}

for i in range(conn_norm_ds.shape[0]):
    data[i] = {}
    tmp_sax = transformer.transform(conn_norm_ds[i, :].reshape(1, -1))
    for j in range(tmp_sax.shape[1]):
        data[i][j] = [tmp_sax[:, j][0], '{}'.format(j), conn_norm_ds[:, i][j]]
        # data[i] = {'letter': list(tmp_sax[:, i]), 'id': [int(x) for x in list(np.array(range(0,tmp_sax.shape[0])))], 'density': list(X[:, i])}

letter_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
letter2num = [0, 1, 2, 3, 4, 5, 6, 7]
for i in range(len(data)):
    for j in range(len(data[i])):
        letter = data[i][j][0]


with open('func.json', 'w') as fp:
    json.dump(data, fp, indent=2, sort_keys=False)

