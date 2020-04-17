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