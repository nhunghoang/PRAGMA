"""
Clustering functions for the timeseries data. 
"""

import numpy as np 
from sklearn.cluster import KMeans, AgglomerativeClustering
import json

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
    
