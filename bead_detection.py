import cv2
import numpy as np
import pandas as pd
import os
import glob


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx <= mn:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - mn) / (mx - mn)
    return (img * 255).astype(np.uint8)

def component_score(img_u8, labels, stats, centroids, idx):
    x = stats[idx, cv2.CC_STAT_LEFT]
    y = stats[idx, cv2.CC_STAT_TOP]
    w = stats[idx, cv2.CC_STAT_WIDTH]
    h = stats[idx, cv2.CC_STAT_HEIGHT]
    area = stats[idx, cv2.CC_STAT_AREA]

    # mask for this component
    comp_mask = (labels[y:y+h, x:x+w] == idx).astype(np.uint8)

    # mean intensity inside blob
    patch = img_u8[y:y+h, x:x+w]
    mean_int = float(patch[comp_mask == 1].mean())

    # approximate circularity using bbox ratio (fast)
    aspect = max(w, h) / max(1, min(w, h))  # 1.0 is best

    # score: prefer bright, compact, roundish blobs
    score = mean_int - 20.0 * (aspect - 1.0)  # tune 20.0 if needed
    return score, mean_int, aspect, area

def detect_beads_single_image(
    image_path: str,
    K: int,
    invert: bool = False,
    min_area: int = 10,
    max_area: int = 2000,
):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return []

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_u8 = to_uint8(img)
    if invert:
        img_u8 = 255 - img_u8

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_u8 = clahe.apply(img_u8)

    # Background removal
    background = cv2.GaussianBlur(img_u8, (0, 0), sigmaX=15, sigmaY=15)
    flat = cv2.subtract(img_u8, background)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blur = cv2.GaussianBlur(flat, (3, 3), 0)

    mask = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=-2
    )

    # Auto-invert if background dominates
    if mask.mean() > 127:
        mask = 255 - mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    candidates = []

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        aspect = max(w, h) / max(1, min(w, h))
        if aspect > 2.0:  # reject cuboid edges
            continue

        cx, cy = centroids[i]

        # score using flat image (bead-enhanced)
        patch = flat[y:y+h, x:x+w]
        comp_mask = (labels[y:y+h, x:x+w] == i)
        mean_int = patch[comp_mask].mean() if comp_mask.any() else 0.0

        score = mean_int - 20.0 * (aspect - 1.0)

        candidates.append((score, float(cx), float(cy), area))

    # ---- THIS IS WHERE K IS USED ----
    candidates.sort(key=lambda t: t[0], reverse=True)
    candidates = candidates[:K]

    beads = [(x, y, area) for (_, x, y, area) in candidates]
    return beads


def build_wide_df_from_folder(
    folder: str,
    K: int = 5,
    pattern: str = "*.png",
    invert: bool = False,
    min_area: int = 10,
    max_area: int = 2000,
):
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise RuntimeError(f"No images found in {folder}")

    rows = []

    for p in paths:
        beads = detect_beads_single_image(
            p, K, invert=invert, min_area=min_area, max_area=max_area
        )

        row = {"image": os.path.basename(p)}

        # Pad if fewer than K
        while len(beads) < K:
            beads.append((np.nan, np.nan, np.nan))

        for i in range(K):
            x, y, area = beads[i]
            row[f"x{i+1}"] = x
            row[f"y{i+1}"] = y
            row[f"area{i+1}"] = area

        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------
# Folder → wide DataFrame
# ----------------------------
if __name__ == "__main__":
    folder = "projections_png_real"
    MAX_BEADS = 5  # set this to expected number of beads

    df = build_wide_df_from_folder(folder, K=MAX_BEADS, pattern="*.png", invert=False, min_area=40, max_area=2000)
    print(df)
