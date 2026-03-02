import napari
from imageio.v3 import imread, imwrite
import numpy as np
from glob import glob
from skimage import feature, transform
from scipy.ndimage import rotate

import os
os.environ["QT_QPA_PLATFORM"] = "windows:fontengine=freetype"

# ----------------------------
# Stack rotate + crop function
# ----------------------------
def rotate_and_crop_stack(stack, angle=90, sigma_thresh=1):
    # Rotate entire stack in (y, x)
    rot_stack = rotate(
        stack,
        angle,
        axes=(2, 1),
        reshape=False,
        order=1,
        mode="constant",
        cval=0,
    )

    # Contrast-based crop (computed once)
    bg = np.median(rot_stack)
    noise = np.std(rot_stack)
    mask = rot_stack > (bg + sigma_thresh * noise)

    if not np.any(mask):
        return rot_stack

    z, y, x = np.where(mask)
    cropped = rot_stack[:, y.min():y.max()+1, x.min():x.max()+1]

    return cropped



# ----------------------------
# Load TIFFs as a stack
# ----------------------------
input_dir = r'C:\Users\Flex Ray\Documents\JGM\AutoCalibration\2026-02-09_Beads_phantom\2026-02-09_Beads_phantom\Scan1\recon'
output_dir = input_dir.replace('recon', 'cropped')

files = sorted(glob(os.path.join(input_dir, '*.tif')))
os.makedirs(output_dir, exist_ok=True)

stack = np.stack([imread(f) for f in files])
print("Original stack shape:", stack.shape)

# ----------------------------
# Rotate + crop once
# ----------------------------
cropped_stack = rotate_and_crop_stack(stack, angle=27, sigma_thresh=2.5)
print("Cropped stack shape:", cropped_stack.shape)

# ----------------------------
# Save cropped slices
# ----------------------------
for f, img in zip(files, cropped_stack):
    out_path = os.path.join(output_dir, os.path.basename(f))
    imwrite(out_path, img)

recon = np.stack([imread(im) for im in sorted(glob('C:\\Users\\Flex Ray\\Documents\\JGM\\AutoCalibration\\2026-02-09_Beads_phantom\\2026-02-09_Beads_phantom\\Scan1\\cropped\\*.tif'))])

napari.imshow(recon, axis_labels=("z", "y", "x"))
napari.run()
