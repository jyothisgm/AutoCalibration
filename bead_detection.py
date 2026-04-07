import os
import glob
import cv2
import numpy as np
import pandas as pd


def detect_beads_single_image(image_path: str, K: int, min_area: int = 10, max_area: int = 2000, exclude_border: int = 0, connectivity: int = 8, tolerance: int = 100):
    """
    Detect up to K bright blobs near the image max value.
    ignore blobs touching border within this margin (px) -> exclude_border
    connectivity: 4 or 8
    Returns a list of (cx, cy, area) sorted by area desc, length <= K.
    """
    # img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # if img is None:
    #     return []

    # if img.ndim == 3:
    #     # print("Converting to grayscale")
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_np = np.asarray(img)
    h, w = img_np.shape[:2]

    max_val = int(img_np.max())
    low = max(0, max_val - int(tolerance))

    # Mask pixels near max
    mask = ((img_np >= low) & (img_np <= max_val)).astype(np.uint8) * 255

    # Connected components on that mask
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)

    cands = []
    # skip background
    skip_blob_count = 0
    for i in range(1, num):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = int(stats[i, cv2.CC_STAT_AREA])

        # ar = bw / max(1, bh)  # aspect ratio
        # if ar > 1.4 or ar < 1/1.4:
        #     continue

        if area < min_area or area > max_area:
            # print(f"  Skipping blob {i} with area {area} outside [{min_area}, {max_area}]")
            skip_blob_count += 1
            continue

        # if exclude_border > 0:
        #     if (x <= exclude_border or y <= exclude_border or
        #         (x + bw) >= (w - exclude_border) or (y + bh) >= (h - exclude_border)):
        #         continue

        cx, cy = centroids[i]
        cands.append((float(cx), float(cy), area, (bh, bw)))
    # if len(cands) != K:
    #     # print(f"Bead areas in {os.path.basename(image_path)}: {[c[3] for c in cands]}")
    #     # print(f"Bead positions in {os.path.basename(image_path)}: {[c[:2] for c in cands]}")
    #     print(f"Found {len(cands)} candidates, needed {K} and skipped {skip_blob_count}")

    # Prefer larger blobs; take top-K
    # cands.sort(key=lambda t: t[2], reverse=True)
    #prefer blobs from top down (since beads are usually near top of image)

    cands.sort(key=lambda t: t[1])
    h, w = img.shape[:2]
    # print(f"Detected {len(beads)} beads in {p}")
    # Pad if fewer than K
    # --- pad to K instead of skipping ---
    if len(cands) < K:
        missing = K - len(cands)

        # Use y-extremes among detected beads
        ys = [b[1] for b in cands]
        xs = [b[0] for b in cands]
        # print(f"Image {os.path.basename(image_path)}: Found {len(cands)} candidates (skipped {skip_blob_count} blobs), max_val={max_val}, low={low}, image shape={img.shape}")
        # print(f"Only found {len(cands)} candidates, need {K}, missing {missing}. Detected ys: {ys}")

        # If no beads at all, choose a deterministic fallback
        if len(cands) == 0:
            # all missing -> pad at top by default (x=0, y=0)
            pad_beads = [(0.0, 0.0, 0.0, None)] * missing
            cands = pad_beads
        else:
            y_top = float(min(ys))
            y_bot = float(max(ys))

            dist_top = y_top - 0.0
            dist_bot = float((h - 1)) - y_bot

            # Use x of the extreme bead as the padded x (keeps it plausible)
            x_top = float(cands[ys.index(min(ys))][0])
            x_bot = float(cands[ys.index(max(ys))][0])

            spacing = (ys[1] - ys[0]) if len(ys) >= 2 else float(h) / K

            if dist_top < dist_bot:
                # pad at top-most pixel
                pad_beads = [(x_top, y_top - spacing, 0.0, None)] * missing
                cands = pad_beads + cands
            else:
                # pad at bottom-most pixel
                pad_beads = [(x_bot, float(y_bot + spacing), 0.0, None)] * missing
                cands = cands + pad_beads
            # print("Padded candidates:", cands)
    return cands[:K]


def detect_all_blobs_with_boxes(image_path: str, min_area: int = 10, max_area: int = 2000,
                                connectivity: int = 8, tolerance: int = 100):
    """
    Returns ALL candidate blobs (after area filter) with bbox info:
    each item: dict with keys: cx, cy, area, x, y, w, h
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return [], None

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_np = np.asarray(img)
    max_val = int(img_np.max())
    low = max(0, max_val - int(tolerance))
    mask = ((img_np >= low) & (img_np <= max_val)).astype(np.uint8) * 255

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)

    blobs = []
    other_blobs = []
    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        cx, cy = centroids[i]

        if area < min_area or area > max_area:
            other_blobs.append({
                "cx": float(cx), "cy": float(cy), "area": area,
                "x": x, "y": y, "w": w, "h": h
            })
            continue

        blobs.append({
            "cx": float(cx), "cy": float(cy), "area": area,
            "x": x, "y": y, "w": w, "h": h
        })

    return blobs, other_blobs, img_np


def save_debug_rectangles(image_path: str, out_dir: str, K: int,
                          min_area: int = 10, max_area: int = 2000,
                          connectivity: int = 8, tolerance: int = 100,
                          thickness: int = 2):
    """
    Saves an annotated image:
      - selected top K (by y ascending) in BLUE
      - remaining candidates in RED
    """
    os.makedirs(out_dir, exist_ok=True)

    blobs, other_blobs, img_np = detect_all_blobs_with_boxes(
        image_path,
        min_area=min_area,
        max_area=max_area,
        connectivity=connectivity,
        tolerance=tolerance
    )
    if img_np is None:
        return False

    # select top K from top->bottom (same rule you use)
    blobs_sorted = sorted(blobs, key=lambda b: b["cy"])
    selected = blobs_sorted[:K]
    selected_ids = set(id(b) for b in selected)

    # draw on a visible 8-bit canvas
    canvas = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # draw non-selected first (red), then selected (blue) on top
    for b in blobs_sorted:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        color = (0, 0, 255)  # RED (BGR)
        if id(b) in selected_ids:
            continue
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

    for b in selected:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        color = (255, 0, 0)  # BLUE (BGR)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

    for b in other_blobs:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        color = (255, 255, 0)  # YELLOW (BGR)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

    out_path = os.path.join(out_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, canvas)
    return True

def build_wide_df_from_folder(folder: str, K: int = 5, min_area: int = 10, max_area: int = 2000, exclude_border: int = 0, connectivity: int = 8, file_type: str =".png", tolerance: int=50, indices: np.ndarray | None = None, box_images: bool=False):
    paths = sorted(entry.path for entry in os.scandir(folder) if entry.is_file() and entry.name.endswith(file_type))
    if not len(paths):
        raise RuntimeError(f"No images found in {folder}")

    rows = []
    # Only create debug dir if needed
    debug_dir = None
    if box_images:
        debug_dir = os.path.join(folder, "debug_rectangles")
        os.makedirs(debug_dir, exist_ok=True)
    if indices is None:
        indices = np.arange(len(paths))
    paths = [paths[i] for i in indices]

    N = len(paths)

    for p in paths:
        if box_images:
            save_debug_rectangles(p, debug_dir, K=K, min_area=min_area, max_area=max_area, connectivity=connectivity, tolerance=tolerance)
        beads = detect_beads_single_image(p, K=K, min_area=min_area, max_area=max_area, exclude_border=exclude_border, connectivity=connectivity, tolerance=tolerance)
        row = {"image": os.path.basename(p)}

        for i in range(K):
            x, y, area, _ = beads[i]
            row[f"x{i+1}"] = x
            row[f"y{i+1}"] = y
            row[f"area{i+1}"] = area
        rows.append(row)

    proj_data = pd.DataFrame(rows)
    # Check that bead detections are consistent across projections (no misidentification)
    range_df = proj_data.select_dtypes(include=np.number).agg(lambda col: col.max() - col.min()).to_frame(name="range").T

    y_cols = [col for col in range_df.columns if col.startswith('y')]
    over_20 = range_df[y_cols] > 30
    if over_20.any(axis=1).any():
        print(range_df)
        print("FAILURE!!!! Some beads might have been misidentified")
    return proj_data

if __name__ == "__main__":
    folder = f"real_scans/2026-02-19_Beads_phantom/Scan1/out_line_integrals"
    folder = f"fake_projections/test"
    K = 5

    df = build_wide_df_from_folder(folder, K=K, min_area=10, max_area=2000, exclude_border=0, connectivity=8, file_type=".png", tolerance=130)
    numeric_cols = df.select_dtypes(include=np.number)

    # Compute range (max - min)
    range_row = numeric_cols.max() - numeric_cols.min()

    # Add as a new row at the end
    df.loc["range"] = range_row
    
    range_row["image"] = "range"
    df.loc[len(df)] = range_row

    df.to_csv(os.path.join(folder, "bead_detections.csv"), index=False)
