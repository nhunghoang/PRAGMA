import hdf5storage
import numpy as np

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
l = len(rois)
pearson = np.round(np.corrcoef(rois), 3)
pearson_matrix = np.reshape(pearson, [l, l])
th_mask = pearson_matrix >= 0.5
pearson_matrix[th_mask == 0 ] = 0

data = []
unique_id = ["%02d" % x for x in range(l)]
for i, p in enumerate(pearson_matrix):
    data.append({'{}'.format(unique_id[i]): list(p)})
pass