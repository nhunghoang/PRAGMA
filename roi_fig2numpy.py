import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting
from PIL import Image
import nibabel as nib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import imageio
# import json
# import base64

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
    # X = X[60:445,188:1350]
    # plt.savefig('tmp.png')
    # plt.close('all')
    # X = imageio.imread('tmp.png')

    delme1 = np.array([[255,255,255,255]]*X.shape[0])
    for col in range(X.shape[1]-1, -1, -1):
        if np.array_equal(X[:,col,:], delme1):
            X = np.concatenate((X[:,0:col,:], X[:, col+1:,:]), axis=1)

    delme1 = np.array([[255,255,255,255]]*X.shape[1])
    for row in range(X.shape[0]-1, -1, -1):
        if np.array_equal(X[row,:,:], delme1):
            X = np.concatenate((X[0:row,:,:], X[row+1:,:,:]), axis=0)

    im = Image.fromarray(X)
    im.save("../atlas_data/current_slice.png")

    plt.figure()
    plt.imshow(X)
    plt.show()


    # convert it onto string
    # json.dumps(X.tolist())  # jsonify

    # numpy to byte array, byte array to numpy
    # t = np.arange(25, dtype=np.float64)
    # s = base64.b64encode(X)
    # r = base64.decodebytes(s)
    # q = np.frombuffer(r, dtype=np.float64)
    #
    # print(np.allclose(q, t))  # np.allclose() compare within a tolerance

    # return s


fatlas = '../data/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'  # Shaefer atlas
mni_path = '/home/bayrakrg//neurdy/d3/server_data/mni_masked.nii.gz'
# fig = plotting.plot_roi(fatlas, cut_coords=(8, -4, 9), black_bg=True, cmap='tab10')

vol = nib.load(fatlas).get_fdata()
mni = nib.load(mni_path).get_fdata()
tri_planar_plot(vol, mni, 44, 37, 45)
