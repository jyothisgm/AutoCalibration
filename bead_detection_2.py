import os
import glob
import cv2
import numpy as np
import pandas as pd


def detect_beads_single_image(image_path: str, K: int, min_area: int = 10, max_area: int = 2000, exclude_border: int = 0, connectivity: int = 8):
    """
    Detect up to K bright blobs near the image max value.
    ignore blobs touching border within this margin (px) -> exclude_border
    connectivity: 4 or 8
    Returns a list of (cx, cy, area) sorted by area desc, length <= K.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return []

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_np = np.asarray(img)
    h, w = img_np.shape[:2]

    max_val = int(img_np.max())
    tol= 100
    low = max(0, max_val - int(tol))

    # Mask pixels near max
    mask = ((img_np >= low) & (img_np <= max_val)).astype(np.uint8) * 255

    # Connected components on that mask
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )

    cands = []
    # skip background
    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < min_area or area > max_area:
            print(f"  Skipping blob {i} with area {area} outside [{min_area}, {max_area}]")
            continue

        if exclude_border > 0:
            if (x <= exclude_border or y <= exclude_border or
                (x + bw) >= (w - exclude_border) or (y + bh) >= (h - exclude_border)):
                continue

        cx, cy = centroids[i]
        cands.append((float(cx), float(cy), area))
    if len(cands) != K:
        print(f"  Found {len(cands)} candidates, needed {K}.")
    
    # Prefer larger blobs; take top-K
    cands.sort(key=lambda t: t[2], reverse=True)
    return cands[:K]

def build_wide_df_from_folder(folder: str, K: int = 5, min_area: int = 10, max_area: int = 2000, exclude_border: int = 0, connectivity: int = 8):
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not paths:
        raise RuntimeError(f"No images found in {folder}")

    rows = []
    for p in paths:
        beads = detect_beads_single_image(p, K=K, min_area=min_area, max_area=max_area, exclude_border=exclude_border, connectivity=connectivity)
        row = {"image": os.path.basename(p)}

        # print(f"Detected {len(beads)} beads in {p}")
        # Pad if fewer than K
        while len(beads) != K:
            print(f"FAILED!! Detected {len(beads)} beads in {p}. Needed {K}.")

        for i in range(K):
            x, y, area = beads[i]
            row[f"x{i+1}"] = x
            row[f"y{i+1}"] = y
            row[f"area{i+1}"] = area
        rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    folder = "projections_png_real"
    K = 5

    df = build_wide_df_from_folder(folder, K=K, min_area=10, max_area=2000, exclude_border=0, connectivity=8)
    print(df)
