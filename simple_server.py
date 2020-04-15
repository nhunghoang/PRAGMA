import os
import flask
from flask import Flask
from flask_cors import CORS

from utils import *

# create Flask app
app = Flask(__name__)
CORS(app)

# timeseries data (400 regions x 171 PCs)
timeseries = load_timeseries_data('reduced_SID173839.txt')

@app.route('/grab_data', methods=['GET','POST'])
def get_signals():
    # grab data from the client
    client_data = flask.request.json
    alg = client_data['alg']
    k = client_data['k']
    X_indices = client_data['X_indices']

    # apply clustering 
    new_clusters = apply_clustering(alg, timeseries, X_indices, k)

    # this is returned to the client - note, it will take data_obj, and convert it to a Javascript object
    data_obj = new_clusters
    return flask.jsonify(data_obj)
#


# execute the application (by default, it should be hosted at localhost:5000, which you will see in the output)
if __name__ == '__main__':
    app.run()
