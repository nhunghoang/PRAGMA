from nilearn import datasets, plotting
import nibabel as nib
import numpy.linalg as npl
import matplotlib.pyplot as plt

from pyts.approximation import SymbolicAggregateApproximation
import numpy as np
import json
import hdf5storage
import scipy.signal as ss

indices = [190]

# # LOAD THE CONN MATRIX AND
# # connectivity matrices all
# mat = hdf5storage.loadmat('/home/bayrakrg/neurdy/d3/conn/processed_yeo_id108828.mat')
# conn = mat['Vp_clean'][0, 0]  #default is the 400 parcellation
# del mat
#
# # normalize the fc conn data
# conn_norm = np.transpose((np.transpose(conn) - np.mean(conn, axis=1)) / np.std(conn, axis=1))

f_atlas = '../data/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'
fu_atlas = nib.load(f_atlas)
fun_atlas = fu_atlas.get_fdata()
mask = np.zeros(fu_atlas.shape)

# create a cluster mask
for idx in indices:
    # print(np.sum(fun_atlas == idx))
    # print(np.sum(mask))
    mask = mask + (fun_atlas == idx)
    # f, (ax0, ax1) = plt.subplots(1, 2)
    # ax0.imshow(fun_atlas[:, :, 30] == idx)
    # ax1.imshow(mask[:, :, 30])
    # plt.show()

# mask structural parcellations
satlas = '../data/mni_icbm152_t1_tal_nlin_asym_09c_seg_ds.nii.gz'
str_atlas = nib.load(satlas)
struct_atlas = str_atlas.get_fdata()
masked = struct_atlas.copy()
masked[mask == 0] = 0

# # Display original and masked images side-by-side
# f, (ax0, ax1) = plt.subplots(1, 2)
# ax0.imshow(struct_atlas[:, :, 40])
# ax1.imshow(masked[:, :, 40])
# plt.show()

# get unique values
unique_labels = np.unique(masked)

# load braincolor csv file
filename = '../data/braincolor.csv'
id_to_name = {}
with open(filename, 'r') as f:
    for line in f.readlines():
        label, name = line.strip().split(',')
        id_to_name[int(label)] = name

data = []
# count number for each masked and structural parcellation
# get percentage for each unique label if non-zero

for u in unique_labels:
    u = int(u)
    if u != 0:
        total = np.sum(struct_atlas == u)
        partial = np.sum(masked == u)
        if partial != 0:
            percent = partial*100/total
        if 80 >= percent >= 7:
            data.append({'unique_id': u, 'unique_name': id_to_name[u], 'percentage': np.round(percent, 2)})

pass

# img_4d = '/home/bayrakrg/neurdy/d3/rfMRI_REST1_LR_hp2000_clean.nii.gz'
# dataset = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
# atlas_filename = dataset.maps

# with open('func.json', 'w') as fp:
#     json.dump(data, fp, indent=2, sort_keys=False)

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
