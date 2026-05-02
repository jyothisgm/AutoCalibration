import os
import re
import cv2
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
import itertools
import imageio.v2 as imageio
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image

from astra_server import AstraServer
from bead_detection import build_wide_df_from_folder

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
np.set_printoptions(suppress=True, precision=8)

HERE = Path(__file__).resolve().parent

OBJ_Y_WORLD = 20.0  # Fixed Unity world y-coordinate for the object stage

PARAM_NAMES = [
    "dSx", "dSy",
    "dOx", "dOy", "dOz",
    "dDx", "dDy", "dDz",
    "alpha",
    "offset_x", "offset_z",
]

# -----------------------------------------------------------------------
# Scan settings parsing (from extras/extract_scan_settings.py)
# -----------------------------------------------------------------------

def _extract_first_float(line):
    return float(re.search(r":\s*([-+]?\d*\.?\d+)", line).group(1))

def _extract_roi(line):
    values = re.search(r":\s*([0-9,\s]+)", line).group(1)
    return np.array([int(v.strip()) for v in values.split(",")], dtype=np.int32)

def parse_scan_settings(file_path):
    params = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("mag_obj"):
                params["mag_obj"] = _extract_first_float(line)
            elif line.startswith("mag_det"):
                params["mag_det"] = _extract_first_float(line)
            elif line.startswith("rot_obj"):
                params["rot_obj"] = _extract_first_float(line)
            elif line.startswith("ver_obj"):
                params["ver_obj"] = _extract_first_float(line)
            elif line.startswith("ver_tube"):
                params["ver_tube"] = _extract_first_float(line)
            elif line.startswith("ver_det"):
                params["ver_det"] = _extract_first_float(line)
            elif line.startswith("tra_obj"):
                params["tra_obj"] = _extract_first_float(line)
            elif line.startswith("tra_tube"):
                params["tra_tube"] = _extract_first_float(line)
            elif line.startswith("tra_det"):
                params["tra_det"] = _extract_first_float(line)
            elif line.startswith("Original pixel size"):
                params["original_pixel_size"] = _extract_first_float(line)
            elif line.startswith("Binning value"):
                params["binning_value"] = int(_extract_first_float(line))
            elif line.startswith("Binned pixel size"):
                params["binned_pixel_size"] = _extract_first_float(line)
            elif line.startswith("ROI (LTRB)"):
                params["ROI"] = _extract_roi(line)
            elif line.startswith("SOD"):
                params["SOD"] = _extract_first_float(line)
            elif line.startswith("SDD"):
                params["SDD"] = _extract_first_float(line)
    return params

def extract_scan_info(scan_root: Path):
    """
    Parse 'scan settings.txt' in scan_root and print a summary of all
    geometry and detector values derived from it.

    Returns a dict with:
        src            – Unity world source position  [x, y, z]
        obj            – Unity world object position  [x, y, z]
        det            – Unity world detector position [x, y, z]
        image_width    – detector width  in pixels (from ROI + binning)
        image_height   – detector height in pixels (from ROI + binning)
        det_spacing    – binned pixel size (mm)
        SOD            – source-object distance  (mm, from file)
        SDD            – source-detector distance (mm, from file)
    """
    settings_path = Path(scan_root) / "scan settings.txt"
    if not settings_path.exists():
        raise FileNotFoundError(f"No 'scan settings.txt' found in {scan_root}")

    p = parse_scan_settings(settings_path)

    roi     = p["ROI"]           # [L, T, R, B]
    binning = p.get("binning_value", 2)
    image_width  = (roi[2] + 1 - roi[0]) // binning
    image_height = (roi[3] + 1 - roi[1]) // binning

    src = np.array([p["tra_tube"], p["ver_tube"], 0.0],          dtype=np.float32)
    obj = np.array([p["tra_obj"],  OBJ_Y_WORLD,   p["mag_obj"]], dtype=np.float32)
    det = np.array([p["tra_det"],  p["ver_det"],  p["mag_det"]], dtype=np.float32)

    det_spacing = p["binned_pixel_size"]
    SOD = p.get("SOD")
    SDD = p.get("SDD")

    print(f"\n{'='*55}")
    print(f"Scan info: {settings_path}")
    print(f"{'='*55}")
    print(f"  src (tube)  : x={src[0]:10.6f}, y={src[1]:10.6f}, z={src[2]:10.6f}")
    print(f"  obj         : x={obj[0]:10.6f}, y={obj[1]:10.6f}, z={obj[2]:10.6f}")
    print(f"  det         : x={det[0]:10.6f}, y={det[1]:10.6f}, z={det[2]:10.6f}")
    print(f"  image_width : {image_width}")
    print(f"  image_height: {image_height}")
    print(f"  det_spacing : {det_spacing:.6f} mm  (binned pixel size)")
    if SOD is not None:
        if SOD is not None:
            print(f"  SOD         : {SOD:.6f} mm")
    if SDD is not None:
        print(f"  SDD         : {SDD:.6f} mm")

    initial_angle_deg = p.get("rot_obj")
    proj_dir = Path(scan_root) / "out_line_integrals"
    projections = len([f for f in proj_dir.iterdir() if f.suffix == ".tif"]) if proj_dir.exists() else None

    if initial_angle_deg is not None:
        print(f"  initial_angle: {initial_angle_deg:.6f} deg")
    print(f"  projections : {projections}")
    print(f"{'='*55}\n")

    return {
        "src":               src,
        "obj":               obj,
        "det":               det,
        "image_width":       image_width,
        "image_height":      image_height,
        "det_spacing":       det_spacing,
        "SOD":               SOD,
        "SDD":               SDD,
        "initial_angle_deg": initial_angle_deg,
        "projections":       projections,
    }

def to_astra_line_integrals(scan_dir, out_dir, eps=1e-6, use_median=False):
    """
    Convert raw scanner projections (dark/flat corrected) to ASTRA-ready
    line integrals and save as .tif in out_dir.
    Mirrors extras/image_flip.py:to_astra_line_integrals.
    """
    os.makedirs(out_dir, exist_ok=True)

    scan_paths = sorted(glob.glob(os.path.join(scan_dir, "scan_*.tif*")))
    di_paths   = sorted(glob.glob(os.path.join(scan_dir, "di*.tif*")))
    io_paths   = sorted(glob.glob(os.path.join(scan_dir, "io*.tif*")))

    if not scan_paths:
        raise RuntimeError(f"No projections scan_*.tif found in {scan_dir}")
    if not di_paths:
        raise RuntimeError(f"No dark field di*.tif found in {scan_dir}")
    if not io_paths:
        raise RuntimeError(f"No flat field io*.tif found in {scan_dir}")

    di_stack = np.stack([imageio.imread(p).astype(np.float32) for p in di_paths], axis=0)
    io_stack = np.stack([imageio.imread(p).astype(np.float32) for p in io_paths], axis=0)

    reducer = np.median if use_median else np.mean
    Id  = reducer(di_stack, axis=0)
    I0c = np.maximum(reducer(io_stack, axis=0) - Id, eps)

    for p in scan_paths:
        I   = imageio.imread(p).astype(np.float32)
        Ic  = np.maximum(I - Id, eps)
        T   = np.clip(Ic / I0c, eps, 1.0)
        proj = (-np.log(T)).astype(np.float32)

        out_path = os.path.join(out_dir, os.path.basename(p))
        img_u8 = apply_napari_contrast_and_gamma(proj, low_percentile=99.5, high_percentile=100.0, gamma=0.2)
        cv2.imwrite(out_path, img_u8)

    print(f"Saved {len(scan_paths)} line-integral images to {out_dir}")

def build_geometry_from_scan_dir(scan_dir: Path, out_line_integrals_subdir="out_line_integrals"):
    settings_path = scan_dir / "scan settings.txt"
    if not settings_path.exists():
        return None

    p = parse_scan_settings(settings_path)

    roi = p["ROI"]   # [L, T, R, B]
    binning = p.get("binning_value", 2)
    image_width  = (roi[2] + 1 - roi[0]) // binning
    image_height = (roi[3] + 1 - roi[1]) // binning

    proj_dir = scan_dir / out_line_integrals_subdir
    if proj_dir.exists():
        projections = len([f for f in proj_dir.iterdir() if f.suffix == ".tif"])
    else:
        projections = None

    return {
        "name":              scan_dir.name,
        "src":               np.array([p["tra_tube"], p["ver_tube"], 0.0],          dtype=np.float32),
        "det":               np.array([p["tra_det"],  p["ver_det"],  p["mag_det"]], dtype=np.float32),
        "obj":               np.array([p["tra_obj"],  OBJ_Y_WORLD,   p["mag_obj"]], dtype=np.float32),
        "initial_angle_deg": p["rot_obj"],
        "projections":       projections,
        "image_width":       image_width,
        "image_height":      image_height,
        "det_spacing":       p["binned_pixel_size"],
    }

def build_geometry_list(scan_root: Path, out_line_integrals_subdir="out_line_integrals"):
    geometries = []
    for scan_dir in sorted(scan_root.iterdir()):
        if not scan_dir.is_dir():
            continue
        geom = build_geometry_from_scan_dir(scan_dir, out_line_integrals_subdir)
        if geom is not None:
            geometries.append(geom)
    return geometries

# -----------------------------------------------------------------------
# Scan settings writer (from extras/scan_settings.py)
# -----------------------------------------------------------------------

def write_scan_settings_txt(
    out_dir,
    image_width,
    image_height,
    voxel_size,
    det_spacing,
    src_world,
    obj_world,
    det_world,
    initial_calibration,
    astra_scaling,
    angles_deg,
):
    start_dt = datetime.now()
    duration_sec = max(1, int(0.1 * len(angles_deg)))
    stop_dt = start_dt + timedelta(seconds=duration_sec)

    src_world = np.asarray(src_world, dtype=np.float64).reshape(3)
    obj_world = np.asarray(obj_world, dtype=np.float64).reshape(3)
    det_world = np.asarray(det_world, dtype=np.float64).reshape(3)

    ic = np.asarray(initial_calibration, dtype=np.float64)
    if ic.shape != (3, 3):
        raise ValueError(f"initial_calibration must have shape (3,3). Got {ic.shape}.")

    src_world = src_world + ic[0]
    obj_world = obj_world + ic[1]
    det_world = det_world + ic[2]

    SOD = float(np.linalg.norm((src_world - obj_world) * astra_scaling))
    SDD = float(np.linalg.norm((src_world - det_world) * astra_scaling))
    magnification = float(SDD / SOD) if SOD > 0 else 0.0

    HC = float(image_width / 2.0)
    VC = float(image_height / 2.0)
    COR = float(obj_world[1])
    voxel_size = det_spacing / magnification * 1000

    start_angle = float(np.min(angles_deg)) if len(angles_deg) else 0.0
    if len(angles_deg) >= 2:
        step = float(angles_deg[1] - angles_deg[0])
        last_angle = 360.0 if abs((float(angles_deg[-1]) + step) - 360.0) < 1e-3 else float(np.max(angles_deg))
    else:
        last_angle = float(np.max(angles_deg)) if len(angles_deg) else 0.0

    def fmt_date(dt): return dt.strftime("%d/%m/%Y")
    def fmt_time(dt): return dt.strftime("%H:%M:%S")
    def fmt_duration(sec):
        mins = sec // 60
        return f"{sec} seconds" if mins <= 1 else f"{mins} minutes"

    scan_id  = start_dt.strftime("%y%m%d_%H%M%S")
    pixel_size = float(det_spacing)

    path = os.path.join(out_dir, "scan settings.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
f"""ScanID : {scan_id}
BatchID :
Project :
START
date: {fmt_date(start_dt)}; time: {fmt_time(start_dt)}

STOP
date: {fmt_date(stop_dt)}; time: {fmt_time(stop_dt)}

SCAN DURATION : {fmt_duration(duration_sec)}
COR : {COR:.6f}
VC : {VC:.6f}
HC : {HC:.6f}
SDD : {SDD:.6f}
SOD : {SOD:.6f}
Voxel size : {voxel_size:.6f}
Magnification : {magnification:.6f}

Operator :
Sample name :
Sample owner :
Application area :
Sample size : 0.000000 mm
Comment :
Scanner name :
Scanner type :
Acquila version :

X-ray tube : xraysource
Tube voltage : 90.000000
Tube power : 49.500000
Vacuum level :
Filter :
Focus mode : microfocus
Target current : 0.000000
Filament status :
Filament mode :
Output mode : Emission current

Camera : detector
Exposure time (ms) : 99.998199
Number of averages : 10.000000
Original pixel size : {pixel_size/2:.6f}
Imaging mode :
Binning value : 2
Binned pixel size : {pixel_size:.6f}
ROI (LTRB) :

Script summary:

scan type:
smooth scan
# projections: {len(angles_deg)}
Start angle : {start_angle:.6f}
Last angle : {last_angle:.6f}
# pre flat fields: 0
# post flat fields: 0
# offset images: 0
Reference images every 0 projections
Preheating time: 0 minutes
Axis for flat field movement :
""")

# -----------------------------------------------------------------------
# Geometry helpers (from phantom_projector.py)
# -----------------------------------------------------------------------

def reset_folder(folder):
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)

def apply_napari_contrast_and_gamma(img, low_percentile=99.0, high_percentile=100.0, gamma=0.2):
    if img is None:
        raise ValueError("Could not read image")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    low  = np.percentile(img, low_percentile)
    high = np.percentile(img, high_percentile)
    if high <= low:
        return np.zeros_like(img, dtype=np.uint8)
    clipped = np.clip(img, low, high)
    norm    = (clipped - low) / (high - low)
    return (np.power(norm, gamma) * 255).astype(np.uint8)

def rotate_y_xz(v, cosY, sinY):
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([x * cosY - z * sinY, y, x * sinY + z * cosY], dtype=np.float32)

def pack_xzy(v):
    return np.array([v[0], v[2], v[1]], dtype=np.float32)

def unpack_xzy(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return np.array([v[0], v[2], v[1]], dtype=np.float64)

def geom12_metrics(g):
    g = np.asarray(g, dtype=float).reshape(12)
    S = unpack_xzy(g[0:3])
    D = unpack_xzy(g[3:6])
    u = unpack_xzy(g[6:9])
    v = unpack_xzy(g[9:12])
    sod = np.linalg.norm(S)
    sdd = np.linalg.norm(D - S)
    mag = sdd / sod
    r_hat = (D - S) / np.linalg.norm(D - S)
    n = np.cross(u, v)
    n_hat = n / np.linalg.norm(n)
    inc_deg = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(r_hat, n_hat))), -1.0, 1.0))))
    return sod, sdd, mag, inc_deg

def print_unity_geometry(src_w, obj_w, det_w, rot_y_deg):
    print("\nUnity geometry (world coordinates):")
    print(f"  Source   : x={src_w[0]:8.3f}, y={src_w[1]:8.3f}, z={src_w[2]:8.3f}")
    print(f"  Object   : x={obj_w[0]:8.3f}, y={obj_w[1]:8.3f}, z={obj_w[2]:8.3f}")
    print(f"  Detector : x={det_w[0]:8.3f}, y={det_w[1]:8.3f}, z={det_w[2]:8.3f}")
    print(f"  Obj rotY : {rot_y_deg:8.4f} deg")

def print_initial_calibration(calib):
    print("\nInitial calibration:")
    print(f"  Source   : x={calib[0][0]:8.3f}, y={calib[0][1]:8.3f}, z={calib[0][2]:8.3f}")
    print(f"  Object   : x={calib[1][0]:8.3f}, y={calib[1][1]:8.3f}, z={calib[1][2]:8.3f}")
    print(f"  Detector : x={calib[2][0]:8.3f}, y={calib[2][1]:8.3f}, z={calib[2][2]:8.3f}")

def print_geometry_vector(geom12):
    g = np.asarray(geom12, dtype=float).reshape(12)
    src, det, u, v = g[0:3], g[3:6], g[6:9], g[9:12]
    print("\nASTRA cone_vec geometry:")
    print(f"  Source   : x={src[0]:8.3f}, z={src[1]:8.3f}, y={src[2]:8.3f}")
    print(f"  Detector : x={det[0]:8.3f}, z={det[1]:8.3f}, y={det[2]:8.3f}")
    print(f"  U vector : x={u[0]:8.4f}, z={u[1]:8.4f}, y={u[2]:8.4f}")
    print(f"  V vector : x={v[0]:8.4f}, z={v[1]:8.4f}, y={v[2]:8.4f}")

def unity_geom12_from_world_coords(
    src_world, obj_world, det_world, initial_calibration,
    obj_rot_y_deg, alpha, astra_scaling, det_spacing,
    det_col, det_row, offset_x=0.0, offset_z=0.0,
):
    obj_rot = np.deg2rad(float(obj_rot_y_deg))
    alpha   = np.deg2rad(float(alpha))
    offset_x_rot = offset_x * np.cos(obj_rot) - offset_z * np.sin(obj_rot)
    offset_z_rot = offset_x * np.sin(obj_rot) + offset_z * np.cos(obj_rot)

    src_world = np.asarray(src_world, dtype=np.float32) + initial_calibration[0]
    obj_world = (np.asarray(obj_world, dtype=np.float32) + initial_calibration[1]
                    + np.array([offset_x_rot, 0.0, offset_z_rot], dtype=np.float32))
    det_world = np.asarray(det_world, dtype=np.float32) + initial_calibration[2]
    det_col   = np.asarray(det_col, dtype=np.float32)
    det_row   = np.asarray(det_row, dtype=np.float32)

    srcPos = (src_world - obj_world) * astra_scaling
    detPos = (det_world - obj_world) * astra_scaling
    srcPos[1] = -srcPos[1]
    detPos[1] = -detPos[1]

    u = det_col * det_spacing
    v = det_row * det_spacing

    cosY = float(np.cos(obj_rot + alpha))
    sinY = float(np.sin(obj_rot + alpha))

    srcPos = rotate_y_xz(srcPos, cosY, sinY)
    detPos = rotate_y_xz(detPos, cosY, sinY)
    u      = rotate_y_xz(u, cosY, sinY)
    v      = rotate_y_xz(v, cosY, sinY)

    return np.concatenate([pack_xzy(srcPos), pack_xzy(detPos), pack_xzy(u), pack_xzy(v)]).astype(np.float32)

def fetch_and_save_projections(
    out_dir, src_world, obj_world, det_world_base,
    alpha, angles_deg, offset_x, offset_z,
    image_height, image_width, initial_calibration,
    astra_scaling, det_spacing, voxel_size,
    det_col, det_row, filename_prefix="proj",
    phantom_name="cuboid_phantom.npy", debug=True, normalize=True,
):
    reset_folder(out_dir)
    rec = np.load(phantom_name)

    if debug:
        write_scan_settings_txt(
            out_dir=out_dir,
            image_width=image_width,
            image_height=image_height,
            voxel_size=voxel_size,
            det_spacing=det_spacing,
            src_world=src_world,
            obj_world=obj_world,
            det_world=det_world_base,
            initial_calibration=initial_calibration,
            astra_scaling=astra_scaling,
            angles_deg=angles_deg,
        )

    server = AstraServer(object=rec, image_width=image_width, image_height=image_height, voxel_size=voxel_size)
    geom12_array = []

    for ry in angles_deg:
        geom12 = unity_geom12_from_world_coords(
            src_world=src_world,
            obj_world=obj_world,
            det_world=det_world_base,
            initial_calibration=initial_calibration,
            obj_rot_y_deg=float(ry),
            alpha=float(alpha),
            astra_scaling=astra_scaling,
            det_spacing=det_spacing,
            det_col=det_col,
            det_row=det_row,
            offset_x=offset_x,
            offset_z=offset_z,
        )
        geom12_array.append(geom12)

    geom12_array = np.asarray(geom12_array, dtype=np.float32)

    if debug:
        sod, sdd, mag, inc_deg = geom12_metrics(geom12_array[0])
        print(f"SOD={sod:8.3f} mm | SDD={sdd:8.3f} mm | Mag={mag:6.3f} | incident={inc_deg:6.4f}°")
        np.set_printoptions(suppress=True)

    imgs = server.generate_stacked_images(geom12_array, normalize=normalize)

    for idx in range(imgs.shape[0]):
        img = apply_napari_contrast_and_gamma(imgs[idx], low_percentile=99.5, high_percentile=100.0, gamma=0.2)
        Image.fromarray(img).save(os.path.join(out_dir, f"{filename_prefix}_{idx:03d}.png"))

    server.close()

# -----------------------------------------------------------------------
# Calibration helpers
# -----------------------------------------------------------------------

def parse_int_list(raw):
    if raw is None:
        return None
    parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    return [int(p) for p in parts] if parts else None

def print_theta_table(theta, iteration):
    dSx, dSy, dOx, dOy, dOz, dDx, dDy, dDz, alpha, offset_x, offset_z = theta
    print("\n" + "=" * 60)
    print(f"Iteration {iteration}")
    print("=" * 60)
    print("Theta offsets (Unity frame):")
    print(f"  Source   : dSx={dSx:8.3f}, dSy={dSy:8.3f}, dSz={0.0:8.3f}  (mm)")
    print(f"  Object   : dOx={dOx:8.3f}, dOy={dOy:8.3f}, dOz={dOz:8.3f}  (mm)")
    print(f"  Object offset: offset_x={offset_x:8.3f}, offset_z={offset_z:8.3f}  (mm)")
    print(f"  Detector : dDx={dDx:8.3f}, dDy={dDy:8.3f}, dDz={dDz:8.3f}  (mm)")
    print(f"  Obj Stage rotY : {alpha:8.3f} deg")
    print("=" * 60)

def make_active_mask(fix_source=False, fix_detector=False, fix_object=False, fix_alpha=False, fix_offset=False):
    mask = np.ones(11, dtype=bool)
    if fix_source:
        mask[0] = mask[1] = False
    if fix_object:
        mask[2] = mask[3] = mask[4] = False
    if fix_detector:
        mask[5] = mask[6] = mask[7] = False
    if fix_alpha:
        mask[8] = False
    if fix_offset:
        mask[9] = mask[10] = False
    return mask

def apply_theta_to_geometry(theta, src_world, obj_world, det_world):
    dSx, dSy, dOx, dOy, dOz, dDx, dDy, dDz, alpha, offset_x, offset_z = theta
    src_w = src_world + np.array([dSx, dSy, 0.0], dtype=np.float32)
    obj_w = obj_world + np.array([dOx, dOy, dOz], dtype=np.float32)
    det_w = det_world + np.array([dDx, dDy, dDz], dtype=np.float32)
    return src_w, obj_w, det_w, alpha, offset_x, offset_z

def match_measured_to_pred(meas, pred, area_weight=1e-3):
    K = pred.shape[0]
    best_perm, best_cost = None, np.inf
    for perm in itertools.permutations(range(K)):
        m    = meas[list(perm)]
        cost = np.sum((m - pred) ** 2)
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
        elif best_perm is None:
            print("WHT?????????????????" + str(m - pred))
    return meas[list(best_perm)]

def residual_from_two_dfs(real_df, pred_df, K, area_weight=1e-3, distance_weight=1.0):
    real_df = real_df.sort_values("image").reset_index(drop=True)
    pred_df = pred_df.sort_values("image").reset_index(drop=True)

    if len(real_df) != len(pred_df):
        raise ValueError(f"real_df has {len(real_df)} rows, pred_df has {len(pred_df)} rows")

    r_list, col_names = [], []

    for i in range(len(real_df)):
        image_id = i + 1
        meas = np.array([[real_df.loc[i, f"x{k+1}"], real_df.loc[i, f"y{k+1}"]] for k in range(K)], dtype=np.float64)
        pred = np.array([[pred_df.loc[i, f"x{k+1}"], pred_df.loc[i, f"y{k+1}"]] for k in range(K)], dtype=np.float64)

        meas_aligned = match_measured_to_pred(meas, pred, area_weight=area_weight)
        diff_pts     = (pred - meas_aligned).copy()

        for k in range(K):
            r_list.append(diff_pts[k, 0]); col_names.append(f"{image_id}_b{k+1}_x")
            r_list.append(diff_pts[k, 1]); col_names.append(f"{image_id}_b{k+1}_y")

        for a in range(K):
            for b in range(a + 1, K):
                if np.any(np.isnan(pred[a])) or np.any(np.isnan(pred[b])):
                    continue
                if np.any(np.isnan(meas_aligned[a])) or np.any(np.isnan(meas_aligned[b])):
                    continue
                d_pred = np.linalg.norm(pred[a] - pred[b])
                d_meas = np.linalg.norm(meas_aligned[a] - meas_aligned[b])
                r_list.append((d_pred - d_meas) * distance_weight)
                col_names.append(f"{image_id}_b{a+1}_b{b+1}")

    return np.array(r_list, dtype=np.float64), col_names

def build_residual_image_based(theta, real_df, angles_deg, cfg, pred_dir, debug=True):
    src_w, obj_w, det_w, alpha, offset_x, offset_z = apply_theta_to_geometry(
        theta=theta,
        src_world=cfg["SRC_WORLD"],
        obj_world=cfg["OBJ_WORLD"],
        det_world=cfg["DET_WORLD"],
    )
    fetch_and_save_projections(
        out_dir=pred_dir,
        src_world=src_w,
        obj_world=obj_w,
        det_world_base=det_w,
        alpha=alpha,
        angles_deg=angles_deg,
        offset_x=offset_x,
        offset_z=offset_z,
        image_height=cfg["det_h"],
        image_width=cfg["det_w"],
        initial_calibration=cfg["initial_calibration"],
        astra_scaling=cfg["astra_scaling"],
        det_spacing=cfg["DET_SPACING"],
        voxel_size=cfg["VOXEL_SIZE"],
        det_col=cfg["DET_COL"],
        det_row=cfg["DET_ROW"],
        filename_prefix="proj",
        phantom_name=HERE / "phantoms/scan2_160x240x498_transposed_rotY180.npy",
        debug=debug,
    )

    pred_df = build_wide_df_from_folder(
        pred_dir,
        K=cfg["K"],
        min_area=cfg.get("min_area", 10),
        max_area=cfg.get("max_area", 2000),
        exclude_border=cfg.get("exclude_border", 0),
        connectivity=cfg.get("connectivity", 8),
        file_type=cfg.get("file_type", ".png"),
        tolerance=cfg.get("tolerance", 130),
        box_images=cfg.get("box_images", False),
    )
    if len(pred_df) != len(real_df):
        print(f"FAILED!!!!!!!!!! pred_df has {len(pred_df)} rows, real_df has {len(real_df)} rows")
        return np.empty((0,), dtype=np.float64)

    return residual_from_two_dfs(real_df, pred_df, cfg["K"])

def numerical_jacobian_image_based(theta, active_mask, real_df, angles_deg, cfg, eps, work_dir):
    r0, cols = build_residual_image_based(theta, real_df, angles_deg, cfg, work_dir / "pred_base", debug=True)

    if len(r0) == 0:
        return None, None
    M = r0.size
    active_idx = np.where(active_mask)[0]
    P = active_idx.size
    J = np.zeros((M, P), dtype=np.float64)

    for col, j in enumerate(active_idx):
        t_p, t_m = theta.copy(), theta.copy()
        t_p[j] += eps[j]; t_m[j] -= eps[j]
        r_p, _ = build_residual_image_based(t_p, real_df, angles_deg, cfg, work_dir / f"pred_p_{j:02d}", debug=False)
        r_m, _ = build_residual_image_based(t_m, real_df, angles_deg, cfg, work_dir / f"pred_m_{j:02d}", debug=False)
        if len(r_p) == 0 or len(r_m) == 0:
            continue
        J[:, col] = (r_p - r_m) / (2.0 * eps[j])

    return r0, J, cols

def lm_solve_image_based(real_df, angles_deg, cfg, n_iters=10, lam=1e-2,
                            fix_source=False, fix_detector=False, fix_object=False,
                            fix_offset=False, work_dir="lm_work"):
    os.makedirs(work_dir, exist_ok=True)
    theta       = np.zeros(11, dtype=np.float64)
    eps         = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01], dtype=np.float64)
    active_mask = make_active_mask(fix_source, fix_detector, fix_object, fix_alpha=False, fix_offset=fix_offset)

    dtheta_norm_hist = []
    stall_count      = 0
    df_r0            = None

    for it in range(200):
        print_theta_table(theta, it)

        src_w, obj_w, det_w, _, _, _ = apply_theta_to_geometry(
            theta, src_world=cfg["SRC_WORLD"], obj_world=cfg["OBJ_WORLD"], det_world=cfg["DET_WORLD"],
        )
        print_unity_geometry(cfg["SRC_WORLD"], cfg["OBJ_WORLD"], cfg["DET_WORLD"], angles_deg[0])
        calib = cfg["initial_calibration"]
        print("\nAfter Applying Calibration: ")
        print_unity_geometry(src_w + calib[0], obj_w + calib[1], det_w + calib[2], angles_deg[0])

        r, J, cols = numerical_jacobian_image_based(theta, active_mask, real_df, angles_deg, cfg, eps, work_dir)
        r1 = np.asarray(r, dtype=np.float64).reshape(-1)

        print(f"Residual vector length: {r1.size}, Jacobian shape: {J.shape}")

        A      = J.T @ J
        g      = J.T @ r1
        dtheta = -np.linalg.solve(A + lam * np.eye(A.shape[0]), g)
        dtheta_full = np.zeros_like(theta)
        dtheta_full[active_mask] = dtheta
        new_theta = theta + dtheta_full

        cost = 0.5 * float(r1 @ r1)

        r_new, cols_new = build_residual_image_based(new_theta, real_df, angles_deg, cfg, os.path.join(work_dir, "pred_trial"))

        if cols_new != cols:
            raise ValueError("Column names/order mismatch between base and trial residuals")

        r2       = np.asarray(r_new, dtype=np.float64).reshape(-1)
        cost_new = 0.5 * float(r2 @ r2)
        df_iter  = pd.DataFrame([r1, r2], columns=cols)
        df_iter.insert(0, "iter", it)
        df_iter.insert(1, "state", ["base", "trial"])
        df_iter.insert(2, "cost", [cost, cost_new])
        if df_r0 is None:
            df_r0 = pd.DataFrame(columns=["iter", "state", "cost"] + cols)
        df_r0 = pd.concat([df_r0, df_iter], ignore_index=True)
        df_r0.to_csv(os.path.join(work_dir, "residual_history.csv"), index=False)

        print(f"\niter {it:02d} cost={cost:.6f} -> {cost_new:.6f} |dtheta|={np.linalg.norm(dtheta):.6e}  lambda={lam:.3e}")

        if cost_new < cost:
            theta = new_theta
            lam   = max(lam / 3.0, 1e-6)
        else:
            lam = min(lam * 5.0, 1e6)

        if np.linalg.norm(dtheta) < 1e-6:
            print("Converged.")
            break

        dtheta_norm_hist.append(dtheta)
        if len(dtheta_norm_hist) >= 5:
            recent_norms = [np.linalg.norm(dn) for dn in dtheta_norm_hist[-5:]]
            if max(recent_norms) - min(recent_norms) < 1e-8:
                stall_count += 1
                if stall_count >= 3:
                    print("Stalled.")
                    break
            else:
                stall_count = 0

    print("\nEstimated theta:")
    for name, v in zip(PARAM_NAMES, theta):
        if name == "alpha":
            print(f"{name:>5s}: {v:+.6e} deg)")
        else:
            print(f"{name:>5s}: {v:+.6f}")
    print("\nFinal cost table:")
    df_r0.to_csv(os.path.join(work_dir, "residual_history.csv"), index=False)
    return theta, f"{np.linalg.norm(dtheta):.6e}", cost_new, it + 1


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main(scan_root, angles, K):
    scan_root     = Path(scan_root)

    DET_ROW    = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    DET_COL    = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    MIN_AREA   = 200
    MAX_AREA   = 6000
    VOXEL_SIZE = 0.1
    astra_scaling = 1

    # ---- Read all geometry from scan settings.txt ----
    info = extract_scan_info(scan_root)

    src          = info["src"]
    obj          = info["obj"]
    det          = info["det"]
    image_width  = info["image_width"]
    image_height = info["image_height"]
    det_spacing  = info["det_spacing"]
    SOD          = info["SOD"]
    SDD          = info["SDD"]
    projections  = info["projections"]
    start_deg    = float(info["initial_angle_deg"])

    if projections is None:
        print(f"out_line_integrals not found — generating from raw projections in {scan_root} ...")
        to_astra_line_integrals(scan_dir=str(scan_root), out_dir=str(scan_root / "out_line_integrals"))
        proj_dir = scan_root / "out_line_integrals"
        projections = len([f for f in proj_dir.iterdir() if f.suffix == ".tif"])

    initial_calibration = np.array([
        np.array([0.0, 0.0, 0.0],             dtype=np.float32),
        np.array([0.0, 0.0, 0.0],             dtype=np.float32),
        np.array([src[0]-det[0], src[1]-det[1], SDD - det[2]], dtype=np.float32),
    ])
    
    print_initial_calibration(initial_calibration)

    cfg = {
        "K":                   K,
        "det_h":               image_height,
        "det_w":               image_width,
        "astra_scaling":       astra_scaling,
        "DET_SPACING":         det_spacing,
        "SRC_WORLD":           src,
        "OBJ_WORLD":           obj,
        "DET_WORLD":           det,
        "VOXEL_SIZE":          VOXEL_SIZE,
        "DET_COL":             DET_COL,
        "DET_ROW":             DET_ROW,
        "min_area":            MIN_AREA,
        "max_area":            MAX_AREA,
        "initial_calibration": initial_calibration,
        "box_images":          True,
    }

    print("\n" + "#" * 80)
    print(f"Running Calibration Projections={projections}  Used={angles}  K={K}")
    print(f"  SOD={SOD}  SDD={SDD}")

    indices           = np.linspace(0, projections - 1, angles, dtype=int)
    real_out_dir      = scan_root / "out_line_integrals"
    projection_angles = np.linspace(start_deg, start_deg + 360.0, angles, endpoint=False)

    real_proj = build_wide_df_from_folder(
        real_out_dir, K=K, min_area=MIN_AREA, max_area=MAX_AREA,
        file_type=".tif", tolerance=130, indices=indices, box_images=True,
    )

    theta_hat, _, _, _ = lm_solve_image_based(
        real_proj, projection_angles, cfg, n_iters=50, lam=1e-2,
        fix_source=True, fix_detector=True, fix_object=False, fix_offset=False,
        work_dir=scan_root / f"fake_projections/{angles}",
    )

    print("Final estimated theta:", theta_hat)
    print("#" * 80 + "\n")
    return theta_hat


# -----------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gauss-Newton calibration runner (scan-settings driven)")
    parser.add_argument(
        "scan_root",
        help="Path to a single scan folder containing 'scan settings.txt' and 'out_line_integrals/'.",
    )
    parser.add_argument(
        "-a", "--angles",
        dest="angles", type=int, default=3,
        help="Number of projections to use (default: 360).",
    )
    parser.add_argument(
        "-k", "--beads",
        dest="K", type=int, default=5,
        help="Number of beads (default: 5).",
    )
    args = parser.parse_args()

    main(scan_root=args.scan_root, angles=args.angles, K=args.K)
