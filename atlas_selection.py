from nilearn import datasets, plotting
import nibabel as nib
import numpy.linalg as npl
import matplotlib.pyplot as plt

from pyts.approximation import SymbolicAggregateApproximation
import numpy as np
import json
import hdf5storage
import scipy.signal as ss

# fatlas = '/home/bayrakrg/neurdy/d3/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'
# satlas = '/home/bayrakrg/neurdy/d3/mni_icbm152_t1_tal_nlin_asym_09c_seg_ds.nii.gz'
# img_4d = '/home/bayrakrg/neurdy/d3/rfMRI_REST1_LR_hp2000_clean.nii.gz'
# dataset = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
# atlas_filename = dataset.maps

# LOAD THE CONN MATRIX AND
# connectivity matrices all
mat = hdf5storage.loadmat('/home/bayrakrg/neurdy/d3/conn/processed_yeo_id108828.mat')
conn = mat['Vp_clean'][0, 0]  #default is the 400 parcellation
del mat

# normalize the fc conn data
conn_norm = np.transpose((np.transpose(conn) - np.mean(conn, axis=1)) / np.std(conn, axis=1))

indices = [86, 108, 111, 114, 119, 160, 166, 169, 171, 172, 182, 184, 185, 191, 194, 195, 196, 197, 198, 243, 316, 318,
           320, 375, 376, 381, 382, 388, 390, 391, 393, 394]
cluster_summed = np.zeros_like(conn_norm[0])
cluster = []
for idx in indices:
    cluster_summed = np.add(cluster_summed, conn_norm[idx])
    cluster.append(conn_norm[idx])
ROI = cluster_summed/len(indices)

conn_matrix = np.vstack(cluster)
time_point = 20
conn_norm_ds = np.transpose(ss.resample(np.transpose(conn_matrix), time_point))

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

for i in range(conn_norm_ds.shape[0]): # ROI x time-point
    tmp_sax = transformer.transform(conn_norm_ds[i, :].reshape(1, -1))
    for j in range(tmp_sax.shape[1]):
        data.append({'time': '{}'.format(j), 'letter': letter_dict[tmp_sax[:, j][0]],  'value': np.round(conn_norm_ds[i, :][j], 3)})

with open('func.json', 'w') as fp:
    json.dump(data, fp, indent=2, sort_keys=False)

# atlas = nib.load(atlas_filename)
# aff = atlas.get_affine()
#
# #from voxel to mm
# real_pt = nib.affines.apply_affine(aff, [0,0,0])
#
# #from mm to voxel
# voxel_pt = nib.affines.apply_affine(npl.inv(aff), real_pt)

# print('Atlas ROIs are located at: %s' % atlas_filename)
#
# plotting.plot_roi(atlas_filename, title="Schaefer")
# plotting.show()
#
# plotting.plot_glass_brain(atlas_filename, threshold=3)


pass