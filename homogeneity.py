import hdf5storage
import numpy as np

# full functional conn data
mat = hdf5storage.loadmat('/home/bayrakrg/neurdy/d3/conn/processed_yeo_id108828.mat')
conn = mat['Vp_clean'][0, 0]  #default is the 400 parcellation
del mat
# normalize by row
conn_norm = np.transpose((np.transpose(conn) - np.mean(conn, axis=1)) / np.mean(conn, axis=1))

# homogeneity
dict2 = [{'region': [86, 108, 111, 114, 119, 160, 166, 169, 171, 172, 182, 184, 185, 191, 194, 195, 196, 197, 198, 243, 316, 318,
          320, 375, 376, 381, 382, 388, 390, 391, 393, 394]}, {'region': [86, 108, 111, 114, 119, 160, 166, 169, 171, 172,
          182, 184, 185, 191, 194, 195, 196, 197, 198, 243, 316, 376]}, {'region': [318]},
        {'region': [381, 382, 388, 390, 391, 393, 394, 320, 375]}]

current = [318]
parent = []
dict = {}
count = 0
for d in dict2:
    dict['current'] = current
    parent = parent + d['region']
    dict['parent'] = parent
    if current != d['region']:
        count += 1
        dict['sibling{}'.format(count)] = d['region']

# parent = []
# dict = []
# count = 0
# dict.append({'name': 'current', 'value': current})
#
# for i, d in enumerate(dict2):
#     parent = parent + d['region']
#     if current != d['region']:
#         count += 1
#         dict.append({'name': 'sibling{}'.format(count),  'value': d['region']})
# dict.append({'name': 'parent', 'value': parent})

data = []
# loop for each key in the dictionary (the number of siblings is changing)
for d in dict:
    roi_idx = dict[d]
    # calculate pearson correlation
    l = len(roi_idx)
    if l > 1:
        pearson = np.round(np.corrcoef(conn_norm[roi_idx]), 3)
        pearson_matrix = np.reshape(pearson, [l, l])
        lower = np.tril(pearson_matrix, k=-1)  # lower triangle (w/o diagonal k=-1)
        data.append({'name': d, 'value': np.round(np.mean(lower[np.tril_indices(l, k=-1)]), 3)})
    else:
        data.append({'name': d, 'value': 1})  # if 

pass