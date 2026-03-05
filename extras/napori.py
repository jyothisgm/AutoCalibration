import napari
from imageio.v3 import imread, imwrite
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# recon = np.stack([imread(im) for im in sorted(glob('C:\\Users\\Flex Ray\\Documents\\JGM\\AutoCalibration\\real_scans\\2026-02-19_Beads_phantom\\Scan1\\out_line_integrals\\*.tif'))])
recon = np.stack([imread(im) for im in sorted(glob(f'C:\\Users\\Flex Ray\\Documents\\JGM\\AutoCalibration\\fake_projections\\test\\*.png'))])

print(recon.shape)
napari.imshow(recon)
napari.run()

# recon = np.stack([imread(im) for im in sorted(glob('C:\\Users\\Flex Ray\\Documents\\JGM\\AutoCalibration\\real_scans\\2026-02-09_Beads_phantom\\Scan1\\cropped\\*.tif'))])

# print(recon.shape)
# napari.imshow(recon, axis_labels=("y", "z", "x"))
# napari.run()

# volume = np.load(r"C:\Users\Flex Ray\Documents\JGM\AutoCalibration\phantoms\scan2_160x240x498.npy")
# z, y, x = volume.shape
# print(f"Volume shape: x={x}, y={y}, z={z}")
# plt.figure(figsize=(12,4))

# plt.subplot(1,3,1)
# axis_limits = (0, 600)

# plt.imshow(volume[28], cmap="gray")
# plt.xlim(axis_limits)
# plt.ylim(axis_limits)

# plt.title("Axial")

# plt.subplot(1,3,2)
# plt.imshow(volume[:, 20, :], cmap="gray")
# plt.xlim(axis_limits)
# plt.ylim(axis_limits)
# plt.title("Coronal")

# plt.subplot(1,3,3)
# plt.imshow(volume[:, :, 30], cmap="gray")
# plt.xlim(axis_limits)
# plt.ylim(axis_limits)
# plt.title("Sagittal")

# plt.tight_layout()
# plt.show()

# print()
