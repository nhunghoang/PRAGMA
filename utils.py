"""
Clustering functions for the timeseries data. 
"""

import numpy as np 
from sklearn.cluster import KMeans, AgglomerativeClustering
import json
import hdf5storage
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.stats as stat
from pyts.approximation import SymbolicAggregateApproximation
import nibabel as nib

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


def prep_data(mat_fname, f_atlas, satlas, filename):
    # full functional conn data
    mat = hdf5storage.loadmat(mat_fname)
    conn = mat['Vp_clean'][0, 0]  # default is the 400 parcellation
    del mat
    # normalize by row
    conn_norm = np.transpose((np.transpose(conn) - np.mean(conn, axis=1)) / np.mean(conn, axis=1))

    # make mask for mapping
    fu_atlas = nib.load(f_atlas)
    fun_atlas = fu_atlas.get_fdata()
    mask = np.zeros(fu_atlas.shape)

    # load structural data
    str_atlas = nib.load(satlas)
    struct_atlas = str_atlas.get_fdata()
    masked = struct_atlas.copy()

    # id to name
    id_to_name = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            label, name = line.strip().split(',')
            id_to_name[int(label)] = name

    return conn_norm, mask, fun_atlas, struct_atlas, masked, id_to_name




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
    children = []
    for i in range(k):
        regions = np_idx[np.where(membership==i)[0]]
        regions = list(map(int, regions))
        children.append({'regions': regions})
    return children


def functional_conn(conn_norm, tree_leaves):
    th = 0.5
    # average cluster members to get ROIs
    rois = []
    for l in range(len(tree_leaves)):
        cluster_summed = np.zeros_like(conn_norm[0])
        cluster = []
        # fetch the data
        indices = tree_leaves[l]
        for idx in indices:
            cluster_summed = np.add(cluster_summed, conn_norm[idx])
        ROI = cluster_summed / len(indices)  # averaged within ROI
        rois.append(ROI)

    # calculate pearson correlation
    pearson = []
    l = len(rois)
    for i in range(l):
        for j in range(l):
            pearson.append(np.round((stat.pearsonr(rois[i], rois[j]))[0], 3))
    pearson_matrix = np.reshape(pearson, [l, l])
    th_mask = pearson_matrix >= th
    pearson_matrix[th_mask == 0] = 0
    pearson_matrix = []
    return pearson_matrix

def sax(conn_norm, indices, time_point):

    cluster_summed = np.zeros_like(conn_norm[0])
    cluster = []
    # fetch the data
    for idx in indices:
        cluster_summed = np.add(cluster_summed, conn_norm[idx])
        cluster.append(conn_norm[idx])
    # ROI = cluster_summed / len(indices)  # averaged within ROI
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
                         'value': np.round(conn_norm_ds[i, :][j], 3) + 1})

    return data  # data is in the format that the observable expecting


def structural_mapping(fun_atlas, mask, struct_atlas, masked, id_to_name, indices):
    # create a cluster mask
    for idx in indices:
        mask = mask + (fun_atlas == idx)

    # mask structural parcellations
    masked[mask == 0] = 0

    # get unique values
    unique_labels = np.unique(masked)

    data = []
    # count number for each masked and structural parcellation
    # get percentage for each unique label if non-zero
    for u in unique_labels:
        u = int(u)
        if u != 0:
            total = np.sum(struct_atlas == u)
            partial = np.sum(masked == u)
            if partial != 0:
                percent = partial * 100 / total
            if percent >= 40:
                data.append({'unique_id': u, 'unique_name': id_to_name[u], 'percentage': np.round(percent, 2)})
    return data


def homogeneity(conn_norm, indices, fam_leaves):
    current = indices
    parent = []
    dict = {}
    count = 0
    for d in fam_leaves:
        parent = parent + d['regions']
        dict['parent'] = parent
        dict['current'] = current
        if current != d['regions']:
            count += 1
            dict['sibling{}'.format(count)] = d['regions']

    data = {}
    # loop for each key in the dictionary (the number of siblings is changing)
    for d in dict:
        roi_idx = dict[d]
        # calculate pearson correlation
        # pearson = []
        l = len(roi_idx)
        # for i in range(l):
        #     for j in range(l):
                # pearson.append(np.round((stat.pearsonr(conn_norm[roi_idx[i]], conn_norm[roi_idx[j]]))[0], 3))
        pearson = np.round(np.corrcoef(conn_norm[roi_idx]), 3)
        pearson_matrix = np.reshape(pearson, [l, l])
        lower = np.tril(pearson_matrix, k=-1)  # lower triangle (w/o diagonal k=-1)
        data[d] = np.round(np.mean(lower[np.tril_indices(l, k=-1)]), 3)
    return data

