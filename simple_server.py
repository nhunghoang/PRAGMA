import flask
from flask import Flask
from flask_cors import CORS

from VAML_project.utils import *
import uuid

# create Flask app
app = Flask(__name__)
CORS(app)

############ DATA ############

# PCA-reduced timeseries data (400 regions x 171 PCs)
reduced_ts = load_reduced_data(os.getcwd() + '/in_data/reduced_SID173839.txt')

# functional conn data
mat_filename = os.getcwd() + '/in_data/processed_yeo_id108828.mat'  # Rubinov conn
# structural mapping data
fatlas = os.getcwd() + '/in_data/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'  # Shaefer atlas
satlas = os.getcwd() + '/in_data/mni_icbm152_t1_tal_nlin_asym_09c_seg_ds.nii.gz'  # SLANT atlas
filename = os.getcwd() + '/in_data/braincolor.csv'  # SLANT labels
mni_template = os.getcwd() + '/in_data/mni_masked.nii.gz'
conn_norm, fun_atlas, struct_atlas, id_to_name, template = prep_data(mat_filename, fatlas, satlas, filename, mni_template)

##############################

@app.route('/grab_data', methods=['GET','POST'])
def get_signals():
    # grab data from the client
    client_data = flask.request.json
    op = client_data['operation']

    if op == 'cluster&fc':
        '''Manipulate the tree and update functional connectivity matrix after expand.'''
        alg = client_data['alg']
        k = client_data['k']
        X_indices = client_data['X_indices']
        tree_leaves = client_data['tree_leaves']
        parent_id = client_data['parent_id']
        new_clusters = apply_clustering(alg, reduced_ts, X_indices, k, parent_id)
        new_tree_leaves = insert_cluster(tree_leaves, new_clusters)
        tri_planar_plot(fun_atlas, template, 44, 37, 45, cmap='tab10')
        func_conn = functional_conn(conn_norm, new_tree_leaves)
        all_data = {'new_clusters': new_clusters, 'func_conn': func_conn}
        data_obj = all_data

    elif op == 'fc':
        '''Calculate the functional connectivity matrix after merge and collapse.'''
        tree_leaves = client_data['tree_leaves']
        tri_planar_plot(fun_atlas, template, 44, 37, 45, cmap='tab10')
        func_conn = functional_conn(conn_norm, tree_leaves)
        data_obj = func_conn

    elif op == 'detail_panel':
        '''Here we calculate the selected node data.'''
        X_indices = client_data['X_indices']
        fam_leaves = client_data['family_leaves']  # this is a dictionary
        sax_data = sax(conn_norm, X_indices, time_point=70)
        struct_data = structural_mapping(fun_atlas, struct_atlas, id_to_name, X_indices)
        homogeneity_data = homogeneity(conn_norm, X_indices, fam_leaves)
        all_data = {'sax': sax_data, 'struct': struct_data, 'homogeneity': homogeneity_data}
        data_obj = all_data

    elif op == 'tree2nii':
        '''Save the output image as nifti.'''
        tree_leaves = client_data['tree_leaves']
        unique_filename = str(uuid.uuid4())
        if not os.path.exists('../out_data'): os.mkdir('../out_data')
        path = os.getcwd() + '/../out_data/' + unique_filename + '.nii'
        atlas = fatlas
        tree2nii(atlas, path, tree_leaves)
        message = 'The current atlas was saved in out_data/'
        data_obj = message

    # this is returned to the client
    # note, it will take data_obj, and convert it to a Javascript object
    return flask.jsonify(data_obj)

# @app.route('/template_shape', methods=['GET','POST'])
# def template_shape():
#     return flask.jsonify([int(template.shape[0]),int(template.shape[1]),int(template.shape[2])])
#
# @app.route('/slice_it', methods=['GET','POST'])
# def slice_it():
#     slices = flask.request.json['slices']
#     tri_planar_plot(fun_atlas, template, slices[0],slices[1],slices[2], cmap='tab10')
#     return flask.jsonify({'noop':0})
#

# execute the application (by default, it should be hosted at localhost:5000, which you will see in the output)
if __name__ == '__main__':
    app.run()