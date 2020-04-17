"""
Clustering functions for the timeseries data. 
"""

import numpy as np 
from sklearn.cluster import KMeans, AgglomerativeClustering
import json
import hdf5storage
import matplotlib.pyplot as plt
import scipy.signal as ss
from pyts.approximation import SymbolicAggregateApproximation

'''
returns numpy array of (PCA'd) timeseries data
400 regions x 171 PCs
'''
def load_reduced_data(in_file):
    return np.genfromtxt(in_file, delimiter='\t', autostrip=True)[:,:171]

'''
[One time run] initialize the starting tree dataset 
in_file: text file containing the PCA-reduced timeseries 
out_file: json file containing timeseries hierarchically grouped into clusters
'''
def init_tree_data(n_leaves, in_file, out_file):
    # read in data, apply k-means
    data = load_timeseries_data(in_file)
    model = KMeans(n_clusters=n_leaves) 
    model.fit(data)
    membership = model.labels_

    # form hierarchical structure
    root = { 'regions': list(range(400)), 'children': [] }
    for i in range(n_leaves):
        regions = list(map(int, np.where(membership==i)[0]))
        root['children'].append({ 'regions': regions })

    # write to json file 
    with open(out_file, 'w') as f:
        json.dump(root, f)

'''
return a children dictionary
children['children'] = list of {'regions': regions}, which each represents a new child
'''
def apply_clustering(algorithm, X, indices, k):
    if algorithm == 'KM':
        model = KMeans(n_clusters=k)
    else: # algorithm == 'AC'
        model = AgglomerativeClustering(n_clusters=k)

    X_subset = X[indices]
    model.fit(X_subset)
    membership = model.labels_

    np_idx = np.array(indices)
    children = {'children':[]}
    for i in range(k):
        regions = np_idx[np.where(membership==i)[0]]
        regions = list(map(int, regions))
        children['children'].append({ 'regions': regions })
    return children
    
def sax(conn_norm, time_point):
    # given indices
    indices = [86, 108, 111, 114, 119, 160, 166, 169, 171, 172, 182, 184, 185, 191, 194, 195, 196, 197, 198, 243, 316,
               318, 320, 375, 376, 381, 382, 388, 390, 391, 393, 394]
    cluster_summed = np.zeros_like(conn_norm[0])
    cluster = []
    for idx in indices:
        cluster_summed = np.add(cluster_summed, conn_norm[idx])
        cluster.append(conn_norm[idx])
    ROI = cluster_summed / len(indices)  # averaged within ROI

    conn_matrix = np.vstack(cluster)
    time_point = 20
    conn_norm_ds = np.transpose(ss.resample(np.transpose(conn_matrix), time_point))

    transformer = SymbolicAggregateApproximation(n_bins=8, strategy='uniform')
    # https://pyts.readthedocs.io/en/stable/modules/approximation.html
    # strategy='uniform': all bins in each sample have identical widths,
    # strategy='quantile': all bins in each sample have the same number of points,
    # strategy='normal': bin edges are quantiles from a standard normal distribution.
    data = []

    # MAKE ONE TIME
    # make a letter dict
    letter_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    letter_dict = {}
    for i, l in enumerate(letter_list):
        letter_dict[l] = i

    for i in range(conn_norm_ds.shape[0]):  # ROI x time-point
        tmp_sax = transformer.transform(conn_norm_ds[i, :].reshape(1, -1))
        for j in range(tmp_sax.shape[1]):
            data.append({'time': '{}'.format(j), 'letter': letter_dict[tmp_sax[:, j][0]],
                         'value': np.round(conn_norm_ds[i, :][j], 3)})

    with open('func.json', 'w') as fp:
        json.dump(data, fp, indent=2, sort_keys=False)
