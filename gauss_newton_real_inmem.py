import os
import argparse
import cv2
import numpy as np
import pandas as pd
import itertools
from pathlib import Path

from astra_server import AstraServer
from bead_detection import build_wide_df_from_folder

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
np.set_printoptions(suppress=True, precision=8)

HERE = Path(__file__).resolve().parent

PARAM_NAMES = [
    "dSx", "dSy",
    "dOx", "dOy", "dOz",
    "dDx", "dDy", "dDz",
    "alpha",
    "offset_x", "offset_z",
]

# -----------------------------------------------------------------------
# Geometry helpers (inlined from gauss_newton_scan_2.py)
# -----------------------------------------------------------------------

def apply_napari_contrast_and_gamma(img, low_percentile=99.0, high_percentile=100.0, gamma=0.2):
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

# -----------------------------------------------------------------------
# Calibration parameter helpers
# -----------------------------------------------------------------------

def parse_int_list(raw: str):
    if raw is None:
        return None
    parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]

def print_theta_table(theta, iteration):
    dSx, dSy, dOx, dOy, dOz, dDx, dDy, dDz, alpha, offset_x, offset_z = theta
    print("\n" + "=" * 60)
    print(f"Iteration {iteration}")
    print("=" * 60)
    print("Theta offsets (Unity frame):")
    print(f"  Source         : dSx={dSx:8.3f}, dSy={dSy:8.3f}, dSz={0.0:8.3f}  (mm)")
    print(f"  Object         : dOx={dOx:8.3f}, dOy={dOy:8.3f}, dOz={dOz:8.3f}  (mm)")
    print(f"  Object offset  : offset_x={offset_x:8.3f}, offset_z={offset_z:8.3f}  (mm)")
    print(f"  Detector       : dDx={dDx:8.3f}, dDy={dDy:8.3f}, dDz={dDz:8.3f}  (mm)")
    print(f"  Obj Stage rotY : {alpha:8.3f} deg")
    print("=" * 60)

def make_active_mask(fix_source=False, fix_detector=False, fix_object=False, fix_alpha=False, fix_offset=False, fix_det_z=False):
    mask = np.ones(11, dtype=bool)
    if fix_source:
        mask[0] = mask[1] = False
    if fix_object:
        mask[2] = mask[3] = mask[4] = False
    if fix_detector:
        mask[5] = mask[6] = mask[7] = False
    if fix_det_z:
        mask[7] = False  # dDz is fixed; Dz is already corrected from SDD in initial_calibration
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

# -----------------------------------------------------------------------
# In-memory bead detection (mirrors detect_beads_single_image from bead_detection.py)
# -----------------------------------------------------------------------

def detect_beads_from_array(img_np: np.ndarray, K: int, min_area: int = 10, max_area: int = 2000,
                             exclude_border: int = 0, connectivity: int = 8, tolerance: int = 100):
    if img_np.ndim == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    img_np = img_np.astype(np.uint8)
    h, w = img_np.shape[:2]

    max_val = int(img_np.max())
    low = max(0, max_val - int(tolerance))
    mask = ((img_np >= low) & (img_np <= max_val)).astype(np.uint8) * 255

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)

    cands = []
    for i in range(1, num):
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        cx, cy = centroids[i]
        cands.append((float(cx), float(cy), area, (bh, bw)))

    cands.sort(key=lambda t: t[1])

    if len(cands) < K:
        missing = K - len(cands)
        ys = [b[1] for b in cands]
        if len(cands) == 0:
            cands = [(0.0, 0.0, 0.0, None)] * missing
        else:
            y_top    = float(min(ys))
            y_bot    = float(max(ys))
            dist_top = y_top
            dist_bot = float(h - 1) - y_bot
            x_top    = float(cands[ys.index(min(ys))][0])
            x_bot    = float(cands[ys.index(max(ys))][0])
            spacing  = (ys[1] - ys[0]) if len(ys) >= 2 else float(h) / K
            if dist_top < dist_bot:
                cands = [(x_top, y_top - spacing, 0.0, None)] * missing + cands
            else:
                cands = cands + [(x_bot, float(y_bot + spacing), 0.0, None)] * missing

    return cands[:K]


def build_wide_df_from_images(imgs: np.ndarray, K: int, min_area: int = 10, max_area: int = 2000,
                               exclude_border: int = 0, connectivity: int = 8, tolerance: int = 130):
    rows = []
    for idx in range(imgs.shape[0]):
        img = apply_napari_contrast_and_gamma(imgs[idx], low_percentile=99.5, high_percentile=100.0, gamma=0.2)
        beads = detect_beads_from_array(img, K=K, min_area=min_area, max_area=max_area,
                                         exclude_border=exclude_border, connectivity=connectivity,
                                         tolerance=tolerance)
        row = {"image": f"proj_{idx:03d}.png"}
        for i in range(K):
            x, y, area, _ = beads[i]
            row[f"x{i+1}"] = x
            row[f"y{i+1}"] = y
            row[f"area{i+1}"] = area
        rows.append(row)

    proj_data = pd.DataFrame(rows)
    y_cols = [f"y{k+1}" for k in range(K)]
    range_vals = proj_data[y_cols].max() - proj_data[y_cols].min()
    if (range_vals > 30).any():
        print("FAILURE!!!! Some beads might have been misidentified")
    return proj_data

# -----------------------------------------------------------------------
# Matching and residual helpers
# -----------------------------------------------------------------------

def match_measured_to_pred(meas: np.ndarray, pred: np.ndarray, area_weight: float = 1e-3):
    K = pred.shape[0]
    best_perm, best_cost = None, np.inf
    for perm in itertools.permutations(range(K)):
        m = meas[list(perm)]
        cost = np.sum((m - pred) ** 2)
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
        elif best_perm is None:
            print("WHT?????????????????" + str(m - pred))
    return meas[list(best_perm)]


def residual_from_two_dfs(real_df, pred_df, K, area_weight: float = 1e-3, distance_weight: float = 1.0):
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

# -----------------------------------------------------------------------
# In-memory projection + residual (no disk I/O for predicted images)
# -----------------------------------------------------------------------

def generate_projections_inmem(theta, angles_deg, cfg, server: AstraServer) -> np.ndarray:
    src_w, obj_w, det_w, alpha, offset_x, offset_z = apply_theta_to_geometry(
        theta=theta,
        src_world=cfg["SRC_WORLD"],
        obj_world=cfg["OBJ_WORLD"],
        det_world=cfg["DET_WORLD"],
    )
    geom12_array = np.array([
        unity_geom12_from_world_coords(
            src_world=src_w,
            obj_world=obj_w,
            det_world=det_w,
            initial_calibration=cfg["initial_calibration"],
            obj_rot_y_deg=float(ry),
            alpha=float(alpha),
            astra_scaling=cfg["astra_scaling"],
            det_spacing=cfg["DET_SPACING"],
            det_col=cfg["DET_COL"],
            det_row=cfg["DET_ROW"],
            offset_x=offset_x,
            offset_z=offset_z,
        )
        for ry in angles_deg
    ], dtype=np.float32)
    return server.generate_stacked_images(geom12_array, normalize=True)


def build_residual_inmem(theta, real_df, angles_deg, cfg, server: AstraServer):
    imgs = generate_projections_inmem(theta, angles_deg, cfg, server)
    pred_df = build_wide_df_from_images(
        imgs, K=cfg["K"],
        min_area=cfg.get("min_area", 10),
        max_area=cfg.get("max_area", 2000),
        exclude_border=cfg.get("exclude_border", 0),
        connectivity=cfg.get("connectivity", 8),
        tolerance=cfg.get("tolerance", 130),
    )
    if len(pred_df) != len(real_df):
        print(f"FAILED: pred_df has {len(pred_df)} rows, real_df has {len(real_df)} rows")
        return np.empty((0,), dtype=np.float64), []
    return residual_from_two_dfs(real_df, pred_df, cfg["K"])

# -----------------------------------------------------------------------
# Combined residual across all geometry scenarios (from gauss_newton_geom.py)
# Each scenario contributes independent projections; residuals are stacked
# into a single vector so the optimizer sees all geometries simultaneously.
# -----------------------------------------------------------------------

def build_combined_residual_inmem(theta, scenarios_data, server: AstraServer):
    r_parts, col_parts = [], []
    for sd in scenarios_data:
        r, cols = build_residual_inmem(theta, sd["real_df"], sd["angles_deg"], sd["cfg"], server)
        if len(r) == 0:
            return np.empty((0,), dtype=np.float64), []
        r_parts.append(r)
        col_parts.extend(f"{sd['name']}_{c}" for c in cols)
    return np.concatenate(r_parts), col_parts


def numerical_jacobian_combined_inmem(theta, active_mask, scenarios_data, eps, server: AstraServer):
    r0, cols = build_combined_residual_inmem(theta, scenarios_data, server)
    if len(r0) == 0:
        return None, None, None

    M = r0.size
    active_idx = np.where(active_mask)[0]
    P = active_idx.size
    J = np.zeros((M, P), dtype=np.float64)

    for col, j in enumerate(active_idx):
        t_p, t_m = theta.copy(), theta.copy()
        t_p[j] += eps[j]; t_m[j] -= eps[j]
        r_p, _ = build_combined_residual_inmem(t_p, scenarios_data, server)
        r_m, _ = build_combined_residual_inmem(t_m, scenarios_data, server)
        if len(r_p) == 0 or len(r_m) == 0:
            continue
        J[:, col] = (r_p - r_m) / (2.0 * eps[j])

    return r0, J, cols


def lm_solve_combined_inmem(scenarios_data, server: AstraServer, n_iters=200, lam=1e-2,
                             fix_source=False, fix_detector=False, fix_object=False,
                             fix_offset=False, fix_det_z=False, work_dir="lm_work"):
    os.makedirs(work_dir, exist_ok=True)
    theta       = np.zeros(11, dtype=np.float64)
    eps         = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01], dtype=np.float64)
    active_mask = make_active_mask(fix_source, fix_detector, fix_object, fix_alpha=False, fix_offset=fix_offset, fix_det_z=fix_det_z)

    dtheta_norm_hist = []
    stall_count      = 0
    df_r0            = None
    dtheta           = np.zeros(int(active_mask.sum()), dtype=np.float64)

    for it in range(200):
        print_theta_table(theta, it)

        sd0 = scenarios_data[0]
        src_w, obj_w, det_w, _, _, _ = apply_theta_to_geometry(
            theta, src_world=sd0["cfg"]["SRC_WORLD"],
            obj_world=sd0["cfg"]["OBJ_WORLD"], det_world=sd0["cfg"]["DET_WORLD"],
        )
        print_unity_geometry(src_w, obj_w, det_w, sd0["angles_deg"][0])
        calib = sd0["cfg"]["initial_calibration"]
        print_initial_calibration(calib)
        print("\nAfter Applying Calibration:")
        print_unity_geometry(src_w + calib[0], obj_w + calib[1], det_w + calib[2], sd0["angles_deg"][0])

        r, J, cols = numerical_jacobian_combined_inmem(theta, active_mask, scenarios_data, eps, server)
        if r is None:
            print("Combined residual failed, aborting.")
            break
        r1 = np.asarray(r, dtype=np.float64).reshape(-1)
        print(f"Combined residual length: {r1.size} ({len(scenarios_data)} scenarios), Jacobian: {J.shape}")

        A = J.T @ J
        g = J.T @ r1
        try:
            dtheta = -np.linalg.solve(A + lam * np.eye(A.shape[0]), g)
        except np.linalg.LinAlgError:
            print(f"  [warn] singular matrix at iter {it}, falling back to lstsq")
            dtheta, _, _, _ = np.linalg.lstsq(A + lam * np.eye(A.shape[0]), -g, rcond=None)

        dtheta_full = np.zeros_like(theta)
        dtheta_full[active_mask] = dtheta
        new_theta = theta + dtheta_full

        cost = 0.5 * float(r1 @ r1)

        r_new, cols_new = build_combined_residual_inmem(new_theta, scenarios_data, server)
        if cols_new != cols:
            raise ValueError("Column names/order mismatch between base and trial residuals")

        r2       = np.asarray(r_new, dtype=np.float64).reshape(-1)
        cost_new = 0.5 * float(r2 @ r2)

        df_iter = pd.DataFrame([r1, r2], columns=cols)
        df_iter.insert(0, "iter",  it)
        df_iter.insert(1, "state", ["base", "trial"])
        df_iter.insert(2, "cost",  [cost, cost_new])
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
    df_r0.to_csv(os.path.join(work_dir, "residual_history.csv"), index=False)
    return theta, f"{np.linalg.norm(dtheta):.6e}", cost_new, it + 1


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gauss-Newton calibration — in-memory projections, combined geometry"
    )
    parser.add_argument(
        "-a", "--angles",
        dest="angles",
        default=None,
        help="Comma/space-separated list of N_ANGLE values. Example: '8,12,24'",
    )
    parser.add_argument(
        "-s", "--scenario",
        dest="scenario",
        default=None,
        help="Comma/space-separated scenario names to include. Example: 'Scan3,Scan4'",
    )
    args = parser.parse_args()

    GEOM_SCENARIOS = [
        {
            'name': 'Scan1',
            'src': np.array([ 0.      , 24.997368,  0.      ], dtype=np.float32),
            'det': np.array([ -25.31836 ,   18.686905, 1059.      ], dtype=np.float32),
            'obj': np.array([  0.540527, 20 , 600.      ], dtype=np.float32),
            'initial_angle_deg': -1.039974,
            'projections': 1434,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan2',
            'src': np.array([ 4.999512, 29.994888,  0.      ], dtype=np.float32),
            'det': np.array([ -25.31836 ,   18.676949, 1059.      ], dtype=np.float32),
            'obj': np.array([  0.540527, 20 , 600.      ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan3',
            'src': np.array([ 4.999512, 29.99237 ,  0.      ], dtype=np.float32),
            'det': np.array([ -25.31836 ,   18.666954, 1059.      ], dtype=np.float32),
            'obj': np.array([  0.540527, 20 , 799.9995  ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan4',
            'src': np.array([-10.000488,  29.997368,   0.      ], dtype=np.float32),
            'det': np.array([ -20.002441,   33.6557  , 1059.      ], dtype=np.float32),
            'obj': np.array([  0.540527, 20 , 799.9995  ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan5',
            'src': np.array([-10.000488,  29.997368,   0.      ], dtype=np.float32),
            'det': np.array([-20.002441,  33.6557  , 959.      ], dtype=np.float32),
            'obj': np.array([  0.540527, 20 , 600.      ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan6',
            'src': np.array([ 0.000488, 29.997368,  0.      ], dtype=np.float32),
            'det': np.array([ -6.002441,  33.6557  , 959.      ], dtype=np.float32),
            'obj': np.array([  0.540527, 20 , 600.      ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan7',
            'src': np.array([ 4.999512, 29.994888,  0.      ], dtype=np.float32),
            'det': np.array([ -25.31836 ,   18.676949, 1059.      ], dtype=np.float32),
            'obj': np.array([  0.540527, 20  , 600.      ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan8',
            'src': np.array([ 4.999512, 29.99237 ,  0.      ], dtype=np.float32),
            'det': np.array([ -25.31836 ,   18.666954, 1059.      ], dtype=np.float32),
            'obj': np.array([  0.540527, 20 , 799.9995  ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan9',
            'src': np.array([-10.000488,  29.997368,   0.      ], dtype=np.float32),
            'det': np.array([ -20.002441,   33.6557  , 1059.      ], dtype=np.float32),
            'obj': np.array([  0.540527, 20 , 799.9995  ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan10',
            'src': np.array([-10.000488,  29.997368,   0.      ], dtype=np.float32),
            'det': np.array([-20.002441,  33.6557  , 959.      ], dtype=np.float32),
            'obj': np.array([  0.540527, 20 , 600.      ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan11',
            'src': np.array([ 0.000488, 29.997368,  0.      ], dtype=np.float32),
            'det': np.array([ -6.002441,  33.6557  , 959.      ], dtype=np.float32),
            'obj': np.array([  0.540527, 20 , 600.      ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
    ]

    astra_scaling = 1
    DET_ROW = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    DET_COL = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    BEAD_COUNT = K = 5
    MIN_AREA   = 200
    MAX_AREA   = 6000
    VOXEL_SIZE = 0.1

    PHANTOM_PATH  = HERE / "phantoms/scan2_160x240x498_transposed_rotY180.npy"
    BASE_REAL_DIR = HERE / "real_scans/2026-02-19_Beads_phantom"

    used_projections = [360]

    if args.angles is not None:
        parsed = parse_int_list(args.angles)
        used_projections = parsed if parsed is not None else used_projections

    if args.scenario is not None:
        selected = set(args.scenario.replace(",", " ").split())
        GEOM_SCENARIOS = [sc for sc in GEOM_SCENARIOS if sc["name"] in selected]

    # All scenarios must share the same image dimensions so a single AstraServer works.
    assert len({(sc["image_width"], sc["image_height"]) for sc in GEOM_SCENARIOS}) == 1, \
        "All scenarios must have the same image dimensions to share one AstraServer."

    phantom    = np.load(PHANTOM_PATH)
    img_width  = GEOM_SCENARIOS[0]["image_width"]
    img_height = GEOM_SCENARIOS[0]["image_height"]

    server = AstraServer(object=phantom, image_width=img_width, image_height=img_height, voxel_size=VOXEL_SIZE)

    try:
        for each_no_projections in used_projections:
            scenario_names_str = "+".join(sc["name"] for sc in GEOM_SCENARIOS)
            print("\n" + "#" * 80)
            print(f"Running scenarios=[{scenario_names_str}] N_ANGLES={each_no_projections}, K={K}")

            scenarios_data = []
            for sc in GEOM_SCENARIOS:
                scenario_name     = sc["name"]
                projections       = sc["projections"]
                indices           = np.linspace(0, projections - 1, each_no_projections, dtype=int)
                real_out_dir      = BASE_REAL_DIR / scenario_name / "out_line_integrals"
                start_deg         = float(sc["initial_angle_deg"])
                projection_angles = np.linspace(start_deg, start_deg + 360.0, each_no_projections, endpoint=False)

                # Compute SDD from source-to-detector distance; use it to fix Dz in initial_calibration.
                # dDx and dDy remain free (optimized); dDz is pre-corrected here and fixed during opt.
                # sdd = float(np.linalg.norm(sc["src"] - sc["det"]))
                initial_calibration = np.array([
                    np.array([0.0, 0.0, 0.0],      dtype=np.float32),
                    np.array([0.0, 0.0, 0.0],      dtype=np.float32),
                    np.array([0.0, 0.0, 40.00080], dtype=np.float32),
                ])
                # print(f"  [{scenario_name}] SDD={sdd:.4f}  det_z={sc['det'][2]:.4f}  Dz_correction={sdd - sc['det'][2]:.4f}")

                real_proj = build_wide_df_from_folder(
                    real_out_dir, K=K, min_area=MIN_AREA, max_area=MAX_AREA,
                    file_type=".tif", tolerance=130, indices=indices, box_images=True,
                )
                print(f"  [{scenario_name}] real_proj rows: {len(real_proj)}")

                cfg = {
                    "K":                   K,
                    "det_h":               sc["image_height"],
                    "det_w":               sc["image_width"],
                    "astra_scaling":       astra_scaling,
                    "DET_SPACING":         sc["det_spacing"],
                    "SRC_WORLD":           sc["src"],
                    "OBJ_WORLD":           sc["obj"],
                    "DET_WORLD":           sc["det"],
                    "VOXEL_SIZE":          VOXEL_SIZE,
                    "DET_COL":             DET_COL,
                    "DET_ROW":             DET_ROW,
                    "min_area":            MIN_AREA,
                    "max_area":            MAX_AREA,
                    "initial_calibration": initial_calibration,
                    "tolerance":           130,
                    "box_images":          True,
                }

                scenarios_data.append({
                    "name":       scenario_name,
                    "real_df":    real_proj,
                    "angles_deg": projection_angles,
                    "cfg":        cfg,
                })

            work_dir = HERE / f"fake_projections/trial_inmem/{each_no_projections}/{scenario_names_str}"
            theta_hat, dtheta, cost, it = lm_solve_combined_inmem(
                scenarios_data, server, n_iters=50, lam=1e-2,
                fix_source=True, fix_detector=False, fix_det_z=True,
                fix_object=False, fix_offset=False,
                work_dir=work_dir,
            )

            os.makedirs(HERE / "theta_log", exist_ok=True)
            theta_TXT = HERE / f"theta_log/theta_hat_inmem_{scenario_names_str}_{each_no_projections}.txt"
            with open(theta_TXT, "a") as ftxt:
                ftxt.write(f"# scenarios=[{scenario_names_str}] N_ANGLES={each_no_projections}, K={K}\n")
                ftxt.write(f"# final_cost={cost:.6f}, iterations={it}, final_dtheta_norm={dtheta}\n")
                ftxt.write("# theta_hat\n")
                ftxt.write(" ".join(f"{v:.3f}" for v in theta_hat) + "\n")

            print("Final estimated theta:", theta_hat)
            print("#" * 80 + "\n")

    finally:
        server.close()
