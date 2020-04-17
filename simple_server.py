import os
import flask
from flask import Flask
from flask_cors import CORS

from utils import *
import hdf5storage
import numpy as np
import nibabel as nib

# create Flask app
app = Flask(__name__)
CORS(app)

############ DATA ############

# PCA-reduced timeseries data (400 regions x 171 PCs)
reduced_ts = load_reduced_data('reduced_SID173839.txt')

# full functional conn data
mat = hdf5storage.loadmat('/home/bayrakrg/neurdy/d3/conn/processed_yeo_id108828.mat')
conn = mat['Vp_clean'][0, 0]  #default is the 400 parcellation
del mat
# normalize by row
conn_norm = np.transpose((np.transpose(conn) - np.mean(conn, axis=1)) / np.mean(conn, axis=1))

# TODO
# structural mapping data
# Shaefer atlas
f_atlas = '/home/bayrakrg/neurdy/d3/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'
fu_atlas = nib.load(f_atlas)
fun_atlas = fu_atlas.get_fdata()
mask = np.zeros(fu_atlas.shape)

# SLANT atlas
satlas = '/home/bayrakrg/neurdy/d3/mni_icbm152_t1_tal_nlin_asym_09c_seg_ds.nii.gz'
str_atlas = nib.load(satlas)
struct_atlas = str_atlas.get_fdata()
masked = struct_atlas.copy()

# SLANT labels
filename = '/home/bayrakrg/neurdy/d3/working_dir/braincolor.csv'
id_to_name = {}
with open(filename, 'r') as f:
    for line in f.readlines():
        label, name = line.strip().split(',')
        id_to_name[int(label)] = name

##############################

@app.route('/grab_data', methods=['GET','POST'])
def get_signals():
    # grab data from the client
    client_data = flask.request.json
    op = client_data['operation']

    if op == 'cluster':
        alg = client_data['alg']
        k = client_data['k']
        X_indices = client_data['X_indices']
        new_clusters = apply_clustering(alg, reduced_ts, X_indices, k)
        data_obj = new_clusters

    elif op == 'detail_panel':
        X_indices = client_data['X_indices']
        sax_data = sax(conn_norm, X_indices, time_point=20)
        struct_data = structural_mapping()
        all_data = {'sax': sax_data, 'struct': struct_data}
        data_obj = all_data

    # TODO
    # elif op == '...':

    # this is returned to the client 
    # note, it will take data_obj, and convert it to a Javascript object
    return flask.jsonify(data_obj)

# execute the application (by default, it should be hosted at localhost:5000, which you will see in the output)
if __name__ == '__main__':
    app.run()
