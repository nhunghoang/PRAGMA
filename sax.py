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
# conn = conn[0:5]
conn_norm = np.transpose((np.transpose(conn) - np.mean(conn, axis=1)) / np.std(conn, axis=1))
time_point = 20
conn_norm_ds = np.transpose(ss.resample(np.transpose(conn_norm), time_point))

transformer = SymbolicAggregateApproximation(n_bins=8, strategy='uniform')
# https://pyts.readthedocs.io/en/stable/modules/approximation.html
# strategy='uniform': all bins in each sample have identical widths,
# strategy='quantile': all bins in each sample have the same number of points,
# strategy='normal': bin edges are quantiles from a standard normal distribution.
data = []

# make a letter dict
letter_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
letter_dict = {}
for i, l in enumerate(letter_list):
    letter_dict[l] = i

for i in range(conn_norm_ds.shape[0]):
    tmp_sax = transformer.transform(conn_norm_ds[i, :].reshape(1, -1))
    for j in range(tmp_sax.shape[1]):
        data.append({'time': '{}'.format(j), 'letter': letter_dict[tmp_sax[:, j][0]],  'value': np.round(conn_norm_ds[i, :][j], 3)})

with open('func.json', 'w') as fp:
    json.dump(data, fp, indent=2, sort_keys=False)

