import os
import flask
from flask import Flask
from flask_cors import CORS

from utils import *
import hdf5storage
import numpy as np

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
# structural coordinate mapping data

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
        sax_data = sax(conn_norm, X_indices, time_point)
        data_obj = new_clusters

    elif op == 'SAX':
        X_indices = client_data['X_indices']
        sax_data = sax(conn_norm, X_indices, time_point)
        data_obj = sax_data

    # TODO
    # elif op == '...':

    # this is returned to the client 
    # note, it will take data_obj, and convert it to a Javascript object
    return flask.jsonify(data_obj)

# execute the application (by default, it should be hosted at localhost:5000, which you will see in the output)
if __name__ == '__main__':
    app.run()
