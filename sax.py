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
conn_norm = np.transpose((np.transpose(conn) - np.mean(conn, axis=1)) / np.std(conn, axis=1))

# given indices
time_point = 20
indices = [86, 108, 111, 114, 119, 160, 166, 169, 171, 172, 182, 184, 185, 191, 194, 195, 196, 197, 198, 243, 316,
           318, 320, 375, 376, 381, 382, 388, 390, 391, 393, 394]

cluster_summed = np.zeros_like(conn_norm[0])
cluster = []
# fetch the data
for idx in indices:
    cluster_summed = np.add(cluster_summed, conn_norm[idx])
    cluster.append(conn_norm[idx])
ROI = cluster_summed / len(indices)  # averaged within ROI

conn_matrix = np.vstack(cluster)
# downsample
conn_norm_ds = np.transpose(ss.resample(np.transpose(conn_matrix), time_point))

# SAX
transformer = SymbolicAggregateApproximation(n_bins=8, strategy='uniform')
# https://pyts.readthedocs.io/en/stable/modules/approximation.html
# strategy='uniform': all bins in each sample have identical widths,
# strategy='quantile': all bins in each sample have the same number of points,
# strategy='normal': bin edges are quantiles from a standard normal distribution.

# MAKE ONE TIME
# make a letter dict
letter_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
letter_dict = {}
for i, l in enumerate(letter_list):
    letter_dict[l] = i

data = []
# hacky solution
for letter in letter_dict:
    for i in range(20):
        data.append({'time': '{}'.format(i), 'letter': letter,
                     'value': 0})
# apply SAX
for i in range(conn_norm_ds.shape[0]):  # ROI x time-point
    tmp_sax = transformer.transform(conn_norm_ds[i, :].reshape(1, -1))
    for j in range(tmp_sax.shape[1]):
        data.append({'time': '{}'.format(j), 'letter': letter_dict[tmp_sax[:, j][0]],
                     'value': np.round(conn_norm_ds[i, :][j], 3)})

with open('func.json', 'w') as fp:
    json.dump(data, fp, indent=2, sort_keys=False)

