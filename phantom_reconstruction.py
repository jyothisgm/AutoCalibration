import napari
import os
import numpy as np

from glob import glob
from imageio.v3 import imread, imwrite
from scipy.ndimage import rotate
from skimage.transform import resize


os.environ["QT_QPA_PLATFORM"] = "windows:fontengine=freetype"

# ----------------------------
# Rotate image or stack
# ----------------------------
def rotate_image(img, angle=27):
    return rotate(
        img,
        angle,
        axes=(0, 1) if img.ndim==2 else (2, 1),  # 2D or 3D
        reshape=False,
        order=1,
        mode="constant",
        cval=0,
    )

# ----------------------------
# Compute crop bbox from contrast
# ----------------------------
def compute_crop_bbox(stack, sigma_thresh=1):
    """
    stack: 3D numpy array (z, y, x)
    Returns bounding box: y0, y1, x0, x1
    """
    bg = np.median(stack)
    noise = np.std(stack)
    mask = stack > (bg + sigma_thresh * noise)

    if not np.any(mask):
        raise ValueError("No content found in stack for cropping")

    z, ys, xs = np.where(mask)
    print("Bounding box dimensions (y0, y1, x0, x1):", ys.min(), ys.max(), xs.min(), xs.max())
    print("Breadth (y) and width (x) of cropped region:", ys.max()-(ys.min()), xs.max()-(xs.min()))
    return ys.min(), ys.max(), xs.min(), xs.max()

# ----------------------------
# Apply bbox to image or stack
# ----------------------------
def crop_with_bbox(img, bbox):
    y0, y1, x0, x1 = bbox
    return img[..., y0:y1, x0:x1] if img.ndim==3 else img[y0:y1, x0:x1]





# ----------------------------
# Paths
# ----------------------------
input_dir = r'C:\Users\Flex Ray\Documents\JGM\AutoCalibration\2026-02-19_Beads_phantom\Scan1\recon'
output_dir = input_dir.replace('recon', 'cropped')
os.makedirs(output_dir, exist_ok=True)

files = sorted(glob(os.path.join(input_dir, '*.tif')))

# ----------------------------
# Step 1: Manually select phantom images into stack
# ----------------------------
first_files = files[33:500]
print(f"Loading {len(first_files)} images for initial stack...")
stack = np.stack([imread(f) for f in first_files])

# ----------------------------
# Step 2: Rotate stack to align phantom
# ----------------------------
rot_stack = rotate_image(stack, angle=-0.7)

# ----------------------------
# Step 3: Compute crop bbox from rotated stack
# ----------------------------
bbox = compute_crop_bbox(rot_stack, sigma_thresh=2.5)

# ----------------------------
# Step 4: Crop first stack
# ----------------------------
cropped_stack = crop_with_bbox(rot_stack, bbox)

# ----------------------------
# Step 5: Save cropped slices
# ----------------------------
for f, img in zip(first_files, cropped_stack):
    out_path = os.path.join(output_dir, os.path.basename(f))
    imwrite(out_path, img)

# ----------------------------
# Step 6: Load remaining images and apply same rotation + crop
# ----------------------------
secondary_files = files[500:-23]
print(len(secondary_files), "remaining images to process with same bbox...")

# ----------------------------
# Step 7: Rotate and crop remaining image, save to output
# ----------------------------
for f in secondary_files:
    img = imread(f)
    rotated = rotate_image(img, angle=-0.7)
    cropped = crop_with_bbox(rotated, bbox)
    out_path = os.path.join(output_dir, os.path.basename(f))
    imwrite(out_path, cropped)

print("Total processed images:", len(first_files) + len(secondary_files))
print("Image shapes after cropping should all be the same:", cropped_stack.shape)

# ----------------------------
# Step 8: Load all cropped images into a single stack and save as .npy
# ----------------------------
vol = np.stack([imread(im) for im in sorted(glob('C:\\Users\\Flex Ray\\Documents\\JGM\\AutoCalibration\\2026-02-19_Beads_phantom\\Scan1\\cropped\\*.tif'))])

# To Visualize in Napari
#napari.imshow(recon, axis_labels=("y", "z", "x"))
#napari.run()

#recon = np.transpose(recon, (0, 1, 2))  # (z, y, x) -> (y, z, x)

# np.save(r'C:\Users\Flex Ray\Documents\JGM\AutoCalibration\recon_cropped_scan_19.npy', recon)
# print("Saved recon stack to recon_cropped.npy with shape", recon.shape)

# # Load (assume Napari order Z,Y,X)
# vol = np.load("recon_cropped_scan_19.npy")              # shape should be (595, 285, 190)

# ----------------------------
# Step 9: Resize to target shape (Z, Y, X) = (498, 240, 160) to match size of the phantom
# ----------------------------
target_shape = (498, 240, 160)          # (Z, Y, X)

shrunk_vol = resize(vol, target_shape, order=1, mode="reflect", anti_aliasing=True, preserve_range=True).astype(vol.dtype)

np.save("scan2_160x240x498.npy", shrunk_vol)
print("Resized volume saved as scan2_160x240x498.npy with shape", shrunk_vol.shape)
