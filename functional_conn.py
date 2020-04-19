import hdf5storage
import numpy as np
import scipy.stats as ss


# full functional conn data
mat = hdf5storage.loadmat('/home/bayrakrg/neurdy/d3/conn/processed_yeo_id108828.mat')
conn = mat['Vp_clean'][0, 0]  #default is the 400 parcellation
del mat
# normalize by row
conn_norm = np.transpose((np.transpose(conn) - np.mean(conn, axis=1)) / np.mean(conn, axis=1))

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
th_mask = pearson_matrix >= 0.5
pearson_matrix[th_mask == 0 ] = 0
# data = {}
# for p, i in enumerate(pearson_matrix):
#     data[i] = [i, pearson_matrix[i]]


# homogeneity
dict2 = [{'region': [86, 108, 111, 114, 119, 160, 166, 169, 171, 172, 182, 184, 185, 191, 194, 195, 196, 197, 198, 243, 316, 318,
          320, 375, 376, 381, 382, 388, 390, 391, 393, 394]}, {'region': [86, 108, 111, 114, 119, 160, 166, 169, 171, 172,
          182, 184, 185, 191, 194, 195, 196, 197, 198, 243, 316, 376]}, {'region': [318, 320, 375]},
        {'region': [381, 382, 388, 390, 391, 393, 394]}]

current = [318, 320, 375]
parent = []
dict = {}
count = 0
for d in dict2:
    dict['current'] = current
    parent = parent + d['region']
    dict['parent'] = parent
    if current != d['region']:
        count += 1
        dict['sibling{}'.format(count)] = d['region']




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
