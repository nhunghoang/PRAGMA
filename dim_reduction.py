"""
Preprocessing script for reducing the dimensions of a 3D image scan. 

Before: voxels x time series 
After: super voxels x time series PCs
"""

import h5py 
import numpy as np 
from sklearn.decomposition import PCA
from time import time
import sys 

#################################################
#################################################

'''
fetch scan from mat file
'''
def read_mat(sample_filename, scan_index, scan_dims):
    # read mat file 
    print('reading mat file...')
    with h5py.File(sample_filename) as f:
        if 'Vp_clean' not in list(f.keys()):
            print('BAD SCAN')
            sys.exit()
        scan_objs = f.get('Vp_clean')
        scan_obj = scan_objs[0][scan_index]
        # dereference hdf5 object
        scan_name = h5py.h5r.get_name(scan_obj, f.id)
        scan = np.array(f[scan_name]).T
        if scan.shape != scan_dims:
            print('BAD SCAN')
            sys.exit()

    # standardize the scan 
    mean = np.mean(scan, axis=1)[:, None]
    #std = np.std(scan, axis=1)[:, None]
    stand_scan = (scan-mean)/mean # according to Mika

    return stand_scan

'''
reduce time series dimension using PCA
'''
def reduce_timeseries(scan):
    # apply PCA
    print("applying PCA...")
    model = PCA()
    model.fit(scan)
    
    # get top PCs that explain >= 90% of variance 
    PCs = model.components_
    pc_explain = np.cumsum(model.explained_variance_ratio_)
    idx = np.where(pc_explain >= 0.9)[0][0]
    top_PCs = PCs[:idx+1,:]
    print("\n{:d} PCs to explain {:.2f} of variance\n".format(idx+1, pc_explain[idx]))

    # transform the scan using these PCs
    transformed_scan = np.matmul(scan, top_PCs.T)
    if transformed_scan.shape != (400, idx+1):
        print('applied PCA wrong; shape: ' + str(transformed_scan.shape))
        sys.exit()
    return transformed_scan

'''
write new scan to text file 
'''
def write_scan(new_scan, fname):
    print('writing new scan to file...')
    s = new_scan.shape
    with open(fname, 'w') as f:
        for i in range(s[0]):
            for j in range(s[1]):
                val = round(new_scan[i,j], 5) # round to 5 decimal places 
                f.write(str(val) + '\t')
            f.write('\n')
    print('finished writing')

#################################################
#################################################

# for debugging 
import random
import matplotlib.pyplot as plt

def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels)/255 for _ in range(3))

def plot_timeseries(scan):
    plt.clf()
    for i in range(400):
        plt.plot(range(1140), scan[i], c=random_color())
    plt.savefig('timeseries_' + subjID + '.png')

#################################################
#################################################

def main():
    subjID = '173839'
    sfile = '/data/rubinov_lab/datasets/hcp/data/' + subjID + '/processed.mat'
    scan = read_mat(sfile, 0, (400,1140)) # 1st RS scan 
    new_scan = reduce_timeseries(scan)
    fname = 'reduced_' + subjID + '.txt'
    write_scan(new_scan, fname)

main() 
