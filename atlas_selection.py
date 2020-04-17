from nilearn import datasets, plotting
import nibabel as nib
import numpy.linalg as npl
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt

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
conn_norm = (conn - np.mean(conn, axis=0)) / np.mean(conn, axis=0) #row-wise

indices = [86, 108, 111, 114, 119, 160, 166, 169, 171, 172, 182, 184, 185, 191, 194, 195, 196, 197, 198, 243, 316, 318,
           320, 375, 376, 381, 382, 388, 390, 391, 393, 394]
cluster = np.zeros_like(conn_norm[0])
for idx in indices:
    cluster = np.add(cluster, conn_norm[idx])
ROI = cluster/len(indices)

# atlas = nib.load(atlas_filename)
# aff = atlas.get_affine()
#
# #from voxel to mm
# real_pt = nib.affines.apply_affine(aff, [0,0,0])
#
# #from mm to voxel
# voxel_pt = nib.affines.apply_affine(npl.inv(aff), real_pt)

print('Atlas ROIs are located at: %s' % atlas_filename)

plotting.plot_roi(atlas_filename, title="Schaefer")
plotting.show()

plotting.plot_glass_brain(atlas_filename, threshold=3)


pass