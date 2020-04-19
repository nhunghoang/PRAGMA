import hdf5storage
import numpy as np
import scipy.stats as ss


# full functional conn data
mat = hdf5storage.loadmat('/home/bayrakrg/neurdy/d3/conn/processed_yeo_id108828.mat')
conn = mat['Vp_clean'][0, 0]  #default is the 400 parcellation
del mat
# normalize by row
conn_norm = np.transpose((np.transpose(conn) - np.mean(conn, axis=1)) / np.mean(conn, axis=1))

# labels
csv_fname = '/home/bayrakrg/neurdy/d3/conn/labels400.csv'
id_to_name = {}
with open(csv_fname, 'r') as f:
    for line in f.readlines():
        label, name = line.strip().split(',')
        id_to_name[int(label)] = name

tree_leaves = [[86, 108, 111, 114, 119, 160, 166, 169, 171, 172, 182, 184, 185, 191, 194, 195, 196, 197, 198, 243, 316],
               [318, 320, 375], [376], [381, 382, 388, 390, 391, 393, 394]]

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
        pearson.append(np.round((ss.pearsonr(rois[i], rois[j]))[0], 3))
pearson_matrix = np.reshape(pearson, [l, l])

# homogeneity
dict = {'parent': [86, 108, 111, 114, 119, 160, 166, 169, 171, 172, 182, 184, 185, 191, 194, 195, 196, 197, 198, 243, 316, 318,
          320, 375, 376, 381, 382, 388, 390, 391, 393, 394], 'current': [86, 108, 111, 114, 119, 160, 166, 169, 171, 172,
          182, 184, 185, 191, 194, 195, 196, 197, 198, 243, 316, 376], 'sibling1': [318, 320, 375],
          'sibling2': [381, 382, 388, 390, 391, 393, 394]}



data = {}
# loop for each key in the dictionary (the number of siblings is changing)
for d in dict:
    roi_idx = dict[d]
    # calculate pearson correlation
    pearson = []
    l = len(roi_idx)
    for i in range(l):
        for j in range(l):
            pearson.append(np.round((ss.pearsonr(conn_norm[roi_idx[i]], conn_norm[roi_idx[j]]))[0], 3))
    pearson_matrix = np.reshape(pearson, [l, l])
    lower = np.tril(pearson_matrix, k=-1)  # lower triangle (w/o diagonal k=-1)
    data[d] = np.round(np.mean(lower[np.tril_indices(l, k=-1)]), 3)

pass
