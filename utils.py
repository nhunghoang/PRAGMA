"""
Clustering functions for the timeseries data. 
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np 
from sklearn.cluster import KMeans, AgglomerativeClustering
import json
import hdf5storage
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.stats as stat
from pyts.approximation import SymbolicAggregateApproximation
import nibabel as nib
from PIL import Image
from matplotlib.colors import ListedColormap
import imageio
import os

'''
returns numpy array of (PCA'd) timeseries data
400 regions x 171 PCs
'''
def load_reduced_data(in_file):
    return np.genfromtxt(in_file, delimiter='\t', autostrip=True)[:,:171]

'''
[One time run] initialize the starting tree dataset 
in_file: text file containing the PCA-reduced timeseries 
out_file: json file containing timeseries hierarchically grouped into clusters
'''
def init_tree_data(n_leaves, in_file, out_file):
    # read in data, apply k-means
    data = load_reduced_data(in_file)
    model = KMeans(n_clusters=n_leaves) 
    model.fit(data)
    membership = model.labels_
    c, counts = np.unique(membership, return_counts=True)

    while np.isin(1, counts):
        print("redo kmeans")
        model = KMeans(n_clusters=n_leaves) 
        model.fit(data)
        membership = model.labels_
        c, counts = np.unique(membership, return_counts=True)
    print(counts)

    # form hierarchical structure
    root = { 'regions': list(range(400)), 'children': [] }
    for i in range(n_leaves):
        regions = list(map(int, np.where(membership==i)[0]))
        root['children'].append({ 'regions': regions })

    # write to json file 
    with open(out_file, 'w') as f:
        json.dump(root, f)

def prep_data(mat_fname, f_atlas, satlas, filename, mni_template):
    # full functional conn data
    mat = hdf5storage.loadmat(mat_fname)
    conn = mat['Vp_clean'][0, 0]  # default is the 400 parcellation
    del mat
    # normalize by row
    conn_norm = np.transpose((np.transpose(conn) - np.mean(conn, axis=1)) / np.mean(conn, axis=1))

    # make mask for mapping
    fu_atlas = nib.load(f_atlas)
    fun_atlas = fu_atlas.get_fdata()

    # load structural data
    str_atlas = nib.load(satlas)
    struct_atlas = str_atlas.get_fdata()

    # load template
    mni = nib.load(mni_template)
    template = mni.get_fdata()

    # id to name
    id_to_name = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            label, name = line.strip().split(',')
            id_to_name[int(label)] = name

    return conn_norm, fun_atlas, struct_atlas, id_to_name, template




'''
return a children dictionary
children['children'] = list of {'regions': regions}, which each represents a new child
'''
def apply_clustering(algorithm, X, indices, k, parent_id):
    if algorithm == 'KM':
        model = KMeans(n_clusters=k)
    else: # algorithm == 'AC'
        model = AgglomerativeClustering(n_clusters=k, linkage='average', affinity='cosine')

    X_subset = X[indices]
    model.fit(X_subset)
    membership = model.labels_

    np_idx = np.array(indices)
    children = []
    for i in range(k):
        regions = np_idx[np.where(membership==i)[0]]
        regions = list(map(int, regions))
        children.append({'regions': regions, 'id': parent_id+str(i)})
    return children


def insert_cluster(tree_leaves, new_clusters):
    '''Remove the parent cluster add its children as new clusters.'''
    one = []
    for n in new_clusters:
        one += list(n['regions'])

    new_tree_leaves = tree_leaves.copy()
    new_tree_leaves.remove(sorted(one))
    for n in new_clusters:
        new_tree_leaves.append(list(n['regions']))

    return new_tree_leaves


def functional_conn(conn_norm, tree_leaves):
    th = 0.7
    # average cluster members to get ROIs
    rois = []
    for l in range(len(tree_leaves)):
        cluster_summed = np.zeros_like(conn_norm[0])
        cluster = []
        # fetch the data
        indices = tree_leaves[l]
        for idx in indices:
            cluster_summed = np.add(cluster_summed, conn_norm[idx])
        ROI = cluster_summed / len(indices)  # averaged within ROI
        rois.append(ROI)

    # calculate pearson correlation
    pearson = []
    l = len(rois)
    for i in range(l):
        for j in range(l):
            pearson.append(np.round((stat.pearsonr(rois[i], rois[j]))[0], 3))
    pearson_matrix = np.reshape(pearson, [l, l])
    th_mask = pearson_matrix >= th
    pearson_matrix[th_mask == 0] = 0
    diag_mask = pearson_matrix == 1
    pearson_matrix[diag_mask == 1] = 0

    # add a unique id for mapping
    data = []
    unique_id = ["%02d" % x for x in range(l)]
    for i, p in enumerate(pearson_matrix):
        data.append({'id': '{}'.format(unique_id[i]), 'value': list(p)})

    return data


def sax(conn_norm, indices, time_point):

    cluster_summed = np.zeros_like(conn_norm[0])
    cluster = []
    # fetch the data
    for idx in indices:
        cluster_summed = np.add(cluster_summed, conn_norm[idx])
        cluster.append(conn_norm[idx])
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
        for i in range(20):
            data['{}_{}'.format(i, letter_dict[letter])] = ({'time': '{}'.format(i), 'letter': str(letter_dict[letter]),
                                                             'value': 0})
    # apply SAX
    for i in range(conn_norm_ds.shape[0]):  # ROI x time-point
        tmp_sax = transformer.transform(conn_norm_ds[i, :].reshape(1, -1))
        for j in range(tmp_sax.shape[1]):
            data['{}_{}'.format(j, letter_dict[tmp_sax[0, j]])]['value'] += 1

    data = list(data.values())

    return data  # data is in the format that the observable expecting


def structural_mapping(fun_atlas, struct_atlas, id_to_name, indices):
    # create a cluster mask
    mask = np.zeros(fun_atlas.shape)
    for idx in indices:
        mask = mask + (fun_atlas == idx)

    # mask structural parcellations
    masked = struct_atlas.copy()
    masked[mask == 0] = 0

    # get unique values
    unique_labels = np.unique(masked)

    data = []
    # count number for each masked and structural parcellation
    # get percentage for each unique label if non-zero
    for u in unique_labels:
        u = int(u)
        if u != 0:
            total = np.sum(struct_atlas == u)
            partial = np.sum(masked == u)
            if partial != 0:
                percent = partial * 100 / total
            if 80 >= percent >= 7:
                data.append({'unique_id': u, 'unique_name': id_to_name[u], 'percentage': np.round(percent, 2)})

    # remove this later
    sorted_data = sorted(data, key=lambda i: i['percentage'], reverse=True)
    if len(sorted_data) > 3:
        return sorted_data[0:4]
    else:
        return sorted_data


def homogeneity(conn_norm, indices, fam_leaves):
    current = indices
    current_id = ""
    parent = []
    parent_id = fam_leaves[0]['id'] # can be solved using children IDs
    dict1 = {}
    count = 0
    for d in fam_leaves:
        parent = parent + d['regions']
        if len(d['id']) < len(parent_id): parent_id = d['id']
        if current != d['regions']:
            count += 1
            dict1['Sibling{}'.format(count)] = {'regions': d['regions'], 'id': d['id']}
        else:
            current_id = d['id']
    dict1['Current'] = {'regions': current, 'id': current_id}
    if len(fam_leaves) > 1:
        dict1['Parent'] = {'regions': parent, 'id': parent_id[:-1]}

    data = []
    # loop for each key in the dictionary (the number of siblings is changing)
    for d in dict1:
        roi_idx = dict1[d]['regions']
        # calculate pearson correlation
        l = len(roi_idx)
        if l > 1:
            pearson = np.round(np.corrcoef(conn_norm[roi_idx]), 3)
            pearson_matrix = np.reshape(pearson, [l, l])
            lower = np.tril(pearson_matrix, k=-1)  # lower triangle (w/o diagonal k=-1)
            data.append({'name': d, 'id': dict1[d]['id'], 'value': np.round(np.mean(lower[np.tril_indices(l, k=-1)]), 3)})
        else:
            data.append({'name': d, 'id': dict1[d]['id'], 'value': 1})

    return data


def tree2nii(atlas, path, tree_leaves):
    img = nib.load(atlas)
    fun_atlas = img.get_fdata()
    masked = np.zeros(img.shape)

    # create a cluster mask
    for i, leaf in enumerate(tree_leaves):
        mask = np.zeros(img.shape)
        for idx in leaf:
            if idx != 0:
                mask = mask + (fun_atlas == idx)

        masked[mask == 1] = i + 1

    new_img = nib.Nifti1Image(masked, img.affine, img.header)
    nib.save(new_img, path)


def tri_planar_plot(parc, template, x, y, z, cmap='tab10'):
    fig, axs = plt.subplots(1,3,figsize=(20, 4))

    parc_mask = parc > 0
    parc_mask = parc_mask.astype(np.float) * 0.85

    gray = plt.get_cmap('gray')
    colors = gray(range(256))
    for i in range(60):
        colors[i,:] = [53/256, 54/256, 58/256, 1.0]
    gray = ListedColormap(colors)

    text_color = 'white'
    bar_color = '#effd5f'
    axs[0].imshow(np.rot90(template[x,:,:]), cmap=gray)
    axs[0].imshow(np.rot90(parc[x, :, :]), cmap=cmap, alpha=np.rot90(parc_mask[x, :, :]))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].plot(range(109), [z]*109, color=bar_color)
    axs[0].plot([y] * 91, range(91), color=bar_color)
    axs[0].text(2, 87, 'x={}'.format(x), fontsize=12, color=text_color)

    axs[1].imshow(np.rot90(template[:, y, :]), cmap=gray)
    axs[1].imshow(np.rot90(parc[:, y, :]), cmap=cmap, alpha=np.rot90(parc_mask[:, y, :]))
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].plot(range(91), [z]*91, color=bar_color)
    axs[1].plot([x] * 91, range(91), color=bar_color)
    axs[1].text(15, 17, 'L', fontsize=12, color=text_color)
    axs[1].text(70, 17, 'R', fontsize=12, color=text_color)
    axs[1].text(2, 87, 'y={}'.format(y), fontsize=12, color=text_color)

    axs[2].imshow(np.rot90(template[:, :, z]), aspect='equal', cmap=gray)
    axs[2].imshow(np.rot90(parc[:, :, z]), aspect='equal', cmap=cmap, alpha=np.rot90(parc_mask[:, :, z]))
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].plot(range(91), [y]*91, color=bar_color)
    axs[2].plot([x] * 109, range(109), color=bar_color)
    axs[2].text(15, 17, 'L', fontsize=12, color=text_color)
    axs[2].text(70, 17, 'R', fontsize=12, color=text_color)
    axs[2].text(2, 105, 'z={}'.format(z), fontsize=12, color=text_color)
    plt.subplots_adjust(wspace=None, hspace=None)

    fig.canvas.draw()
    X = np.array(fig.canvas.renderer.buffer_rgba())

    delme1 = np.array([[255,255,255,255]]*X.shape[0])
    for col in range(X.shape[1]-1, -1, -1):
        if np.array_equal(X[:,col,:], delme1):
            X = np.concatenate((X[:,0:col,:], X[:, col+1:,:]), axis=1)

    delme1 = np.array([[255,255,255,255]]*X.shape[1])
    for row in range(X.shape[0]-1, -1, -1):
        if np.array_equal(X[row,:,:], delme1):
            X = np.concatenate((X[0:row,:,:], X[row+1:,:,:]), axis=0)

    im = Image.fromarray(X)
    if not os.path.exists('../atlas_data'): os.mkdir('../atlas_data')
    im.save("../atlas_data/current_slice.png")

