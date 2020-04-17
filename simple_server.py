import os
import flask
from flask import Flask
from flask_cors import CORS

from utils import *

# create Flask app
app = Flask(__name__)
CORS(app)

############ DATA ############

# PCA-reduced timeseries data (400 regions x 171 PCs)
reduced_ts = load_reduced_data('reduced_SID173839.txt')

# TODO
# complete timeseries data 

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
        data_obj = new_clusters

    # TODO
    # elif op == '...':

    # this is returned to the client 
    # note, it will take data_obj, and convert it to a Javascript object
    return flask.jsonify(data_obj)

# execute the application (by default, it should be hosted at localhost:5000, which you will see in the output)
if __name__ == '__main__':
    app.run()