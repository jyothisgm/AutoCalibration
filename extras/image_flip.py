import cv2
import glob, os
import numpy as np
import imageio.v2 as imageio
from pathlib import Path


def read_stack(paths):
    imgs = [imageio.imread(p).astype(np.float32) for p in paths]
    return np.stack(imgs, axis=0)

def apply_napari_contrast_and_gamma(
    image_path: str,
    out_path: str,
    low_percentile: float = 99.0,   # lower contrast limit
    high_percentile: float = 100.0, # upper contrast limit (usually max)
    gamma: float = 0.2
):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not read image")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)

    # ---- Compute contrast limits like napari ----
    low = np.percentile(img, low_percentile)
    high = np.percentile(img, high_percentile)

    if high <= low:
        raise ValueError("Invalid contrast limits")

    # ---- Clip + rescale (napari behavior) ----
    clipped = np.clip(img, low, high)
    norm = (clipped - low) / (high - low)

    # ---- Gamma correction ----
    gamma_corrected = np.power(norm, gamma)

    # Convert to 8-bit for saving
    out = (gamma_corrected * 255).astype(np.uint8)

    cv2.imwrite(out_path, out)

    return low, high


def to_astra_line_integrals(scan_dir, out_dir, eps=1e-6, use_median=False):
    os.makedirs(out_dir, exist_ok=True)
    org_dir = os.path.join(out_dir, "original")
    os.makedirs(org_dir, exist_ok=True)
    scan_paths = sorted(glob.glob(os.path.join(scan_dir, "scan_*.tif*")))
    di_paths   = sorted(glob.glob(os.path.join(scan_dir, "di*.tif*")))
    io_paths   = sorted(glob.glob(os.path.join(scan_dir, "io*.tif*")))

    if not scan_paths:
        raise RuntimeError("No Projections scan_*.tif found")
    if not di_paths:
        raise RuntimeError("No Dark Field di*.tif found")
    if not io_paths:
        raise RuntimeError("No Flat Field io*.tif found")

    di_stack = read_stack(di_paths)
    io_stack = read_stack(io_paths)

    reducer = np.median if use_median else np.mean
    Id = reducer(di_stack, axis=0)
    I0 = reducer(io_stack, axis=0)

    I0c = I0 - Id
    I0c = np.maximum(I0c, eps)

    for p in scan_paths:
        I = imageio.imread(p).astype(np.float32)

        Ic = I - Id
        Ic = np.maximum(Ic, eps)

        T = Ic / I0c
        T = np.clip(T, eps, 1.0)

        proj = -np.log(T).astype(np.float32)

        out_path = os.path.join(org_dir, os.path.basename(p))
        second_out = os.path.join(out_dir, os.path.basename(p))
        imageio.imwrite(out_path, proj)
        apply_napari_contrast_and_gamma(out_path, second_out, low_percentile=99.5, high_percentile=100.0, gamma=0.2)


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent.parent
    DIR = HERE / "real_scans" / "2026-02-19_Beads_phantom"
    for i in range(3, 12):
        CUR_DIR = DIR / f"Scan{i}"
        to_astra_line_integrals(CUR_DIR, CUR_DIR / "out_line_integrals")

