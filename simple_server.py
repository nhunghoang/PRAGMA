import flask
from flask import Flask
from flask_cors import CORS
from utils import *
import globe

# save pid of current python process
with open('server_pids.txt', 'a') as f: f.write( '{}\n'.format(os.getpid()) )

# create Flask app
app = Flask(__name__)
CORS(app)

############ DATA ############

# functional conn data
mat_filename = os.getcwd() + '/in_data/processed_yeo_id108828.mat'  # Rubinov conn

# for the overlay on MNI slices
mni = nib.load(os.getcwd() + '/in_data/mni_masked.nii.gz')
template = mni.get_fdata()

##############################
@app.route('/grab_data', methods=['GET','POST'])
def get_signals():
    # grab data from the client
    client_data = flask.request.json
    op = client_data['operation']

    if op == 'init_tree':
        n = client_data['n']
        pearson_thres = client_data['pearson_thres']
        prep_data(int(n), mat_filename)
        clusters = init_tree(globe.CONN_NORM, pearson_thres, int(n))
        homogeneity_data = homogeneity(clusters)
        color_list = get_network_color_list()
        data_obj = {'init_clusters': clusters, 'homogeneity': homogeneity_data, 'net_colors': color_list}

    if op == 'cluster':
        pearson_thres = client_data['pearson_thres']
        X_indices = client_data['X_indices']
        parent_id = client_data['parent_id']

        # update tree
        new_clusters = apply_clustering(globe.CONN_NORM, X_indices, pearson_thres, parent_id)

        all_data = {'new_clusters': new_clusters}
        data_obj = all_data

    elif op == 'homog':
        tree_data = client_data['tree_data']

        homogeneity_data = homogeneity(tree_data)

        all_data = {'homogeneity': homogeneity_data}
        data_obj = all_data

    elif op == 'detail_panel':
        '''Here we calculate the selected node data.'''
        X_indices = client_data['X_indices']
        globe.CURRENT_ID = client_data['current_id']
        tri_planar_plot(template, globe.X, globe.Y, globe.Z)
        sim1 = sax(X_indices, time_point=70)
        sim2 = md_std(X_indices)
        all_data = {'sax': sim1, 'md_std': sim2}
        data_obj = all_data

    elif op == 'tree2nii':
        '''Save the output image as nifti.'''
        globe.TREE_LEAVES = client_data['tree_leaves']
        globe.TREE_LEAF_IDS = client_data['tree_leaf_ids']
        # print('op_nii', tree_leaves)

        tree2nii(globe.AFFINE, globe.HEADER, op)
        message = 'The current atlas was saved in out_data/'
        data_obj = message

    # this is returned to the client
    # note, it will take data_obj, and convert it to a Javascript object
    return flask.jsonify(data_obj)

@app.route('/template_shape', methods=['GET','POST'])
def template_shape():
    return flask.jsonify([int(globe.X),int(globe.Y),int(globe.Z)])

@app.route('/slice_it', methods=['GET','POST'])
def slice_it():
    slices = flask.request.json['slices']
    globe.X = slices[0]; globe.Y = slices[1]; globe.Z = slices[2]
    tri_planar_plot(template, globe.X, globe.Y, globe.Z)
    return flask.jsonify({'noop':0})

@app.route('/threshold_it', methods=['GET','POST'])
def threshold_it():
    th = flask.request.json['th']
    globe.TREE_LEAVES = flask.request.json['tree_leaves']
    globe.TREE_LEAF_IDS = flask.request.json['tree_leaf_ids']
    globe.CURRENT_ID = flask.request.json['current_id']
    func_conn, new_leafID_order = functional_conn(th)
    tree2nii(globe.AFFINE, globe.HEADER, op=None)
    tri_planar_plot(template, globe.X, globe.Y, globe.Z)
    return flask.jsonify({'func_conn': func_conn, 'fc_leaf_ids': new_leafID_order, 'slices': [globe.X, globe.Y, globe.Z]})


if __name__ == '__main__':
    app.run()
