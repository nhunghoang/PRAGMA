import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting

fatlas = '../data/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'  # Shaefer atlas
fig = plotting.plot_roi(fatlas, cut_coords=(8, -4, 9), black_bg=True, cmap='tab10')

figure = plt.savefig(fig, dpi=300)

# Now we can save it to a numpy array.
data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))