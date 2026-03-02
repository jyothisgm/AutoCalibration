import napari
from imageio.v3 import imread, imwrite
import numpy as np
from glob import glob

recon = np.stack([imread(im) for im in sorted(glob('C:\\Users\\Flex Ray\\Documents\\JGM\\AutoCalibration\\2026-02-19_Beads_phantom\\Scan3\\out_line_integrals\\*.tif'))])

print(recon.shape)
napari.imshow(recon, axis_labels=("z", "y", "x"))
napari.run()

recon = np.stack([imread(im) for im in sorted(glob('C:\\Users\\Flex Ray\\Documents\\JGM\\AutoCalibration\\2026-02-09_Beads_phantom\\Scan1\\cropped\\*.tif'))])

print(recon.shape)
napari.imshow(recon, axis_labels=("z", "y", "x"))
napari.run()

