"""
Clustering functions for the timeseries data. 
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
from nilearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
import hdf5storage
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.stats as sp
from pyts.approximation import SymbolicAggregateApproximation
import nibabel as nib
from PIL import Image
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import os
import uuid
import imageio
from numpy import sign, zeros
from scipy.interpolate import interp1d
import sys
import globe
from skimage import measure
from scipy import signal
import pickle as pkl

#############################################################

# networks, corresponding ids and colors

NETWORKS = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
COLORS = ['#e6194b', '#ff4d00', '#c8ec69', '#3cb44b', '#46f0f0', '#4363d8', '#9966cc']

NET_INDEX = {net:idx for idx,net in enumerate(NETWORKS)}
NET_COLOR = {net:col for net,col in zip(NETWORKS,COLORS)}

#############################################################

def get_model(algorithm, k):
    if algorithm == 'ke':
        model = KMeans(n_clusters=k)
    elif algorithm == 'ac':
        model = AgglomerativeClustering(n_clusters=k, linkage='average', affinity='cosine')
    elif algorithm == 'ae':
        model = AgglomerativeClustering(n_clusters=k, linkage='average', affinity='euclidean')
    elif algorithm == 'we':
        model = AgglomerativeClustering(n_clusters=k, linkage='ward', affinity='euclidean')
    elif algorithm == 'sc':
        model = SpectralClustering(n_clusters=k, affinity='precomputed', n_init=100, assign_labels='discretize')

    return model


def get_network_color_list():
    return [NET_COLOR[net] for net in NETWORKS]

def init_tree(data, pearson_thres, n):
    #TODO remove this band-aid
    algorithm = 'ke'
    
    # split data into initial networks
    left_networks = {} # key: network name, value: list of region indices 
    right_networks = {}
    for i in range(data.shape[0]):
        netw = globe.LABEL_DICT[i]['Network']
        hemi = globe.LABEL_DICT[i]['Hemisphere']
        key = hemi + '_' + netw
        if hemi == 'LH':
            if key not in left_networks.keys(): left_networks[key] = [i]
            else: left_networks[key].append(i)
        elif hemi == 'RH':
            if key not in right_networks.keys(): right_networks[key] = [i]
            else: right_networks[key].append(i)
        else: 
            print('ERROR: hemisphere label is incorrect')

    # left and right networks
    root_children = []
    for i, networks in enumerate([left_networks, right_networks]):
        children = [] 
        regions = []
        for netw in networks.keys():
            net_id = str(NET_INDEX[netw.split('_')[1]])
            regions.extend(networks[netw])
            # agglomerative clustering
            net_data = data[networks[netw]]
            n_clusters, membership = agg_clustering(net_data, pearson_thres)
            np_netw_idx = np.array(networks[netw])
            if n_clusters == 1: 
                # then this network itself is a starting leaf node
                id_ = '/'.join(('0', str(i), net_id))
                children.append({'regions': networks[netw], 'id': id_})
            else:
                # tree data structure for the children
                netw_children = []
                for c in range(n_clusters):
                    c_regions = list(map(int, np_netw_idx[np.where(membership==c)]))
                    id_ = '/'.join(('0', str(i), net_id, str(c)))
                    netw_children.append({'regions': c_regions, 'id': id_})
                id_ = '/'.join(('0', str(i), net_id))
                children.append({'regions': networks[netw], 'id': id_, 'children': netw_children})
        root_children.append({'regions': regions, 'id': '/'.join(('0', str(i))), 'children': children})

    # switch list order of left & right networks, so they show up on the tree as left & right
    root_children.reverse()
    # reverse the left network children so it visually lines up with the right network children
    root_children[1]['children'].reverse()

    # tree data structure for the root
    root = { 'regions': list(range(n)), 'children': root_children, 'id': '0' }

    return root

def agg_clustering(data, pearson_min):
    dist_mat = 1 - np.corrcoef(data)
    model = AgglomerativeClustering(affinity = 'precomputed', 
                                    distance_threshold = 1-pearson_min, 
                                    linkage = 'complete', 
                                    n_clusters = None)
    model.fit(dist_mat) 
    return model.n_clusters_, model.labels_

def optimal_kmeans(data, alg, inertia_thres=0.05):
    k_range = range(1,data.shape[0]+1)
    old_inertia = 1000
    old_labels = None
    old_k = None
    for k in k_range:
        model = get_model(alg, k)
        model.fit(data)
        diff = old_inertia - model.inertia_
        if (diff <= inertia_thres) and (old_labels is not None): 
            return old_k, old_labels
        old_inertia = model.inertia_
        old_labels = model.labels_
        old_k = k
    print('ERROR: models did not reach inertia threshold')
    sys.exit()

def prep_data(n, mat_fname):
    # which Schaefer parcellation fetch and save to in_data
    atlas_info = datasets.fetch_atlas_schaefer_2018(n_rois=n, yeo_networks=7, resolution_mm=2,
                                                    data_dir=os.getcwd() + '/in_data/',
                                                    base_url=None, resume=True, verbose=1)

    # load the parcellation
    parc = nib.load(atlas_info['maps'])
    globe.PARCELLATION = parc.get_fdata()
    globe.ATLAS = parc.get_fdata()
    globe.AFFINE = parc.affine
    globe.HEADER = parc.header
    globe.LABELS = atlas_info['labels']

    # create a dictionary from labels provided by Schaefer atlas
    globe.LABEL_DICT = {} # key: region index, value: dictionary of label info
    for id_, label in enumerate(globe.LABELS):
        label_parts = str(label).strip("b'").split('_')
        if len(label_parts) == 4:
            globe.LABEL_DICT[id_] = {'Hemisphere': label_parts[1], 'Network': label_parts[2], 'Partition': label_parts[3]}
        elif len(label_parts) == 5:
            globe.LABEL_DICT[id_] = {'Hemisphere': label_parts[1], 'Network': label_parts[2], 'Partition': label_parts[4]}
        else:
            print('Error in FC function.')

    # full functional conn data
    mat = hdf5storage.loadmat(mat_fname, variable_names=['Vp_clean'])

    N = int((n / 100) + 2)
    conn = mat['Vp_clean'][0, N]
    del mat
    # normalize by row
    conn_mean = np.mean(conn, axis=1) 
    conn_mean = np.reshape(conn_mean, (conn_mean.shape[0],1))
    globe.CONN_NORM = (conn-conn_mean)/conn_mean
    #globe.CONN_NORM = np.transpose((np.transpose(conn) - np.mean(conn, axis=1)) / np.mean(conn, axis=1))

'''
return a children dictionary
children['children'] = list of {'regions': regions}, which each represents a new child
'''
def apply_clustering(X, indices, pearson_thres, parent_id):
    X_subset = X[indices]
    n_clusters, membership = agg_clustering(X_subset, pearson_thres)

    np_idx = np.array(indices)
    children = []
    for i in range(n_clusters):
        regions = np_idx[np.where(membership==i)[0]]
        regions = list(map(int, regions))
        children.append({'regions': regions, 'id': parent_id + '/' + str(i)})
    return children


def functional_conn(th):
    # average cluster members to get ROIs
    rois = []
    for l in range(len(globe.TREE_LEAVES)):
        cluster_summed = np.zeros_like(globe.CONN_NORM[0])
        cluster = []
        # fetch the data
        indices = globe.TREE_LEAVES[l]
        for idx in indices:
            cluster_summed = np.add(cluster_summed, globe.CONN_NORM[idx])
        ROI = cluster_summed / len(indices)  # averaged within ROI
        rois.append(ROI)

    l = len(rois)
    pearson = np.round(np.corrcoef(rois), 3)
    whole_matrix = np.reshape(pearson, [l, l])
    th_mask = (whole_matrix >= th[0]) & (whole_matrix <= th[1])
    pearson_matrix = whole_matrix.copy()
    pearson_matrix[th_mask == 0] = 0
    diag_mask = pearson_matrix == 1
    pearson_matrix[diag_mask == 1] = 0
    pearson_matrix = np.abs(pearson_matrix)

    hemisphere = ['LH','RH']

    # add a unique id for mapping
    data = []
    for i, p in enumerate(pearson_matrix):
        id_parts = globe.TREE_LEAF_IDS[i].split('/')
        data.append({'id': '{}'.format(globe.TREE_LEAF_IDS[i]), 'value': p, 'Network': '{}'.format(NETWORKS[int(id_parts[2])]),
                     'label': '{}_{}'.format(NETWORKS[int(id_parts[2])], hemisphere[int(id_parts[1])])})

    # creates a key function for sorting
    def dict_idx(x):
        hemisphere = ['LH','RH']
        xnet, xhem = x['label'].split('_')
        xnetidx = NET_INDEX[xnet]
        xhemidx = hemisphere.index(xhem)
        return xnetidx * 2 + xhemidx

    # need to sort rows and cols in the same order 
    sort_key = [dict_idx(dp) for dp in data]
    order = np.argsort(sort_key)
    # print('ORDER', order)
    for dp in data:
        dp['value'] = list(dp['value'][order])
    data = list(np.array(data)[order])

    # order of tree leaf ids after sort
    ordered_leaf_ids = [d['id'] for d in data]

    return data, ordered_leaf_ids


def md_std(indices):
    cluster_summed = np.zeros_like(globe.CONN_NORM[0])
    cluster = []
    # fetch the data
    for idx in indices:
        cluster_summed = np.add(cluster_summed, globe.CONN_NORM[idx])
        cluster.append(globe.CONN_NORM[idx])

    mean_cluster = np.mean(cluster, axis=0)
    median_cluster = np.median(cluster, axis=0)
    if len(indices) > 1:
        se_cluster = sp.sem(cluster, axis=0)
    else:
        se_cluster = np.zeros(len(cluster[0]))

    mean_cluster = (mean_cluster - np.min(mean_cluster)) / np.ptp(mean_cluster)
    median_cluster = (median_cluster - np.min(median_cluster)) / np.ptp(median_cluster)
    if np.min(se_cluster) != 0:
        se_cluster = (se_cluster - np.min(se_cluster)) / np.ptp(se_cluster)

    if np.sum(se_cluster) == 0:
        area_range = [np.min(mean_cluster), np.max(mean_cluster)]
    else:
        area_range = [np.min(se_cluster-mean_cluster), np.max(se_cluster+mean_cluster)]

    data = {}
    unique_id = ["%04d" % x for x in range(len(mean_cluster))]
    for t in range(len(mean_cluster)):
        data[unique_id[t]] = {'time': '{}'.format(t), 'median': median_cluster[t], 'mean': mean_cluster[t],
                              'range': area_range, 'stde': se_cluster[t]}

    data = list(data.values())
    return data


def sax(indices, time_point):
    cluster_summed = np.zeros_like(globe.CONN_NORM[0])
    cluster = []
    # fetch the data
    for idx in indices:
        cluster_summed = np.add(cluster_summed, globe.CONN_NORM[idx])
        cluster.append(globe.CONN_NORM[idx])
    # ROI = cluster_summed / len(indices)  # averaged within ROI
    conn_matrix = np.vstack(cluster)

    # downsample
    conn_norm_ds = np.transpose(ss.resample(np.transpose(conn_matrix), time_point))

    # SAX
    transformer = SymbolicAggregateApproximation(n_bins=8, strategy='uniform')

    # MAKE ONE TIME
    # make a letter dict
    letter_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    letter_dict = {}
    for i, l in enumerate(letter_list):
        letter_dict[l] = i

    data = {}
    # initialize the sax dictionary solution
    for letter in letter_dict:
        for i in range(70):
            data['{}_{}'.format(i, letter_dict[letter])] = ({'time': '{}'.format(i), 'letter': str(letter_dict[letter]),
                                                             'value': 0})
    # apply SAX
    for i in range(conn_norm_ds.shape[0]):  # ROI x time-point
        tmp_sax = transformer.transform(conn_norm_ds[i, :].reshape(1, -1))
        for j in range(tmp_sax.shape[1]):
            data['{}_{}'.format(j, letter_dict[tmp_sax[0, j]])]['value'] += 1

    data = list(data.values())

    return data  # data is in the format that the observable expecting


def homogeneity(tree_data):
    homog_dict = {}

    def homog(node, homog_dict):
        hom = pearson_fcn(node['regions'], node['id'])
        homog_dict[node['id']] = hom[0]['value']
        if 'children' in node.keys():
            for child in node['children']:
                homog(child, homog_dict)

    def pearson_fcn(roi_idx, d):
        data = []
        # calculate pearson correlation
        l = len(roi_idx)
        if l > 1:
            pearson = np.round(np.corrcoef(globe.CONN_NORM[roi_idx]), 3)
            pearson_matrix = np.reshape(pearson, [l, l])
            lower = np.tril(pearson_matrix, k=-1)  # lower triangle (w/o diagonal k=-1)
            data.append({'name': d, 'value': np.round(np.mean(lower[np.tril_indices(l, k=-1)]), 3)})
        else:
            data.append({'name': d, 'value': 1})  # if

        return data

    homog(tree_data, homog_dict)

    return homog_dict


def tree2nii(affine, header, op):
    masked = np.zeros(globe.ATLAS.shape)
    net_index = {'0': 'Vis', '1': 'SomMot', '2': 'DorsAttn', '3': 'SalVentAttn', '4': 'Limbic', '5': 'Cont',
                 '6': 'Default'}
    cmap = []
    # create a cluster mask
    for i, leaf in enumerate(globe.TREE_LEAVES):
        mask = np.zeros(globe.ATLAS.shape)
        for idx in leaf:
            mask = mask + (globe.ATLAS == idx + 1)

        masked[mask == 1] = i + 1
        cmap.append(NET_COLOR[net_index[globe.TREE_LEAF_IDS[i].split('/')[2]]])
    globe.PARCELLATION = masked
    globe.CMAP = cmap

    if op == 'tree2nii':
        unique_filename = str(uuid.uuid4())
        if not os.path.exists('../out_data'): os.mkdir('../out_data')
        path = os.getcwd() + '/../out_data/' + unique_filename + '.nii'
        new_img = nib.Nifti1Image(masked, affine, header)
        nib.save(new_img, path)


def hex2rgb(h):
    h = h.strip('#')
    return [int(h[i:i + 2], 16)/256 for i in (0, 2, 4)]


def get_edges(slice):
    labels = np.unique(slice)
    edges = np.zeros(slice.shape)

    for label in labels:
        if label > 0:
            mask = slice == label

            edge = measure.find_contours(mask, 0.8)
            a = signal.convolve2d(mask, [[1,1,1],[1,1,1],[1,1,1]], mode='same')
            edge = np.multiply(a>0, a<9)

            edges[edge] = 1

    edges[slice==0] = 0
    return edges

def rgbslice(slice, colors, edge):
    vals = np.unique(slice)
    rgb = np.zeros((slice.shape[0], slice.shape[1], 3))

    for i in range(len(vals)):
        rgb[slice==vals[i]] = colors[int(vals[i])]

    return np.concatenate((rgb, np.expand_dims(edge, axis=2)), axis=2)

def tri_planar_plot(template, x_slice, y_slice, z_slice):
    if not (globe.PARCELLATION is None):
        plt.close('all')

        x_slice = x_slice - 1
        y_slice = y_slice - 1
        z_slice = z_slice - 1

        fig, axs = plt.subplots(3,1,figsize=(4, 20))
        reCMAP = globe.CMAP.copy()
        # localize and highlight slice
        if str(globe.CURRENT_ID) == '0':
            reCMAP = globe.CMAP.copy()
        else:
            for leaf_id in globe.TREE_LEAF_IDS:
                current_length = len(globe.CURRENT_ID)
                if globe.CURRENT_ID == leaf_id[:current_length]:
                    current_IDx = globe.TREE_LEAF_IDS.index(str(leaf_id))

                    # print(globe.TREE_LEAVES[current_IDx])
                    reCMAP[current_IDx] = '#000000'

        gray = plt.get_cmap('gray')
        colors = gray(range(256))
        for i in range(60):
            colors[i, :] = [53 / 256, 54 / 256, 58 / 256, 1.0]
        gray = ListedColormap(colors)

        wb_color = ['#ffffff']
        wb_color.extend(reCMAP)

        cmap = [hex2rgb(h) for h in wb_color]

        # cmap = ListedColormap(cmap)
        # norm = Normalize(vmin=0.0, vmax=len(wb_color))

        with open('debug.pkl', 'wb') as f:
            pkl.dump([globe.PARCELLATION, wb_color, globe.TREE_LEAF_IDS, globe.TREE_LEAVES], f)



        text_color = 'white'
        bar_color = '#effd5f'
        axs[0].imshow(np.rot90(template[template.shape[0]-x_slice, :, :]), cmap=gray)
        edge = get_edges(globe.PARCELLATION[template.shape[0]-x_slice, :, :])
        slice = rgbslice(globe.PARCELLATION[template.shape[0] - x_slice, :, :], cmap, edge)
        axs[0].imshow(np.rot90(slice))
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].plot(range(109), [template.shape[2]-z_slice] * 109, color=bar_color)
        axs[0].plot([y_slice] * 91, range(91), color=bar_color)
        axs[0].text(2, 87, 'x={}'.format(x_slice+1), fontsize=12, color=text_color)


        axs[1].imshow(np.flip(np.rot90(template[:, y_slice, :]),1), cmap=gray)
        edge = get_edges(globe.PARCELLATION[:, y_slice, :])
        slice = rgbslice(globe.PARCELLATION[:, y_slice, :], cmap, edge)
        axs[1].imshow(np.flip(np.rot90(slice),1))
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].plot(range(91), [template.shape[2]-z_slice] * 91, color=bar_color)
        axs[1].plot([x_slice] * 91, range(91), color=bar_color)
        axs[1].text(15, 17, 'L', fontsize=12, color=text_color)
        axs[1].text(70, 17, 'R', fontsize=12, color=text_color)
        axs[1].text(2, 87, 'y={}'.format(y_slice+1), fontsize=12, color=text_color)


        axs[2].imshow(np.flip(np.rot90(template[:, :, z_slice]),1), aspect='equal', cmap=gray)
        edge = get_edges(globe.PARCELLATION[:, :, z_slice])
        slice = rgbslice(globe.PARCELLATION[:, :, z_slice], cmap, edge)
        axs[2].imshow(np.flip(np.rot90(slice),1), aspect='equal')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[2].plot(range(91), [template.shape[1]-y_slice]*91, color=bar_color)
        axs[2].plot([x_slice] * 109, range(109), color=bar_color)
        axs[2].text(15, 17, 'L', fontsize=12, color=text_color)
        axs[2].text(70, 17, 'R', fontsize=12, color=text_color)
        axs[2].text(2, 105, 'z={}'.format(z_slice+1), fontsize=12, color=text_color)
        plt.subplots_adjust(wspace=None, hspace=None)

        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())

        delme1 = np.array([[255, 255, 255, 255]] * X.shape[0])
        for col in range(X.shape[1] - 1, -1, -1):
            if np.array_equal(X[:, col, :], delme1):
                X = np.concatenate((X[:, 0:col, :], X[:, col + 1:, :]), axis=1)

        delme2 = np.array([[255, 255, 255, 255]] * X.shape[1])
        for row in range(X.shape[0] - 1, -1, -1):
            if np.array_equal(X[row, :, :], delme2):
                X = np.concatenate((X[0:row, :, :], X[row + 1:, :, :]), axis=0)

        im = Image.fromarray(X)
        im.save("current_slice.png")
    else:
        print('Initializing ...')


