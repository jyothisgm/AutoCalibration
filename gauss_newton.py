import os
import glob
from re import I
from bead_detection_2 import build_wide_df_from_folder
import cv2
import csv
import numpy as np
import pandas as pd
import itertools

from phantom_generator import generate_k_bead_phantom
from phantom_projection import fetch_and_save_projections, print_geometry_vector, print_unity_geometry, unity_geom12_from_worldcoords, unpack_xzy

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


PARAM_NAMES = [
    "dSx", "dSy",
    "dOx", "dOy", "dOz",
    "dDx", "dDy", "dDz",
    "alpha",
]

BOUNDS = {
    "dSx":   (-5.0,  +5.0),
    "dSy":   (-5.0,  +5.0),

    "dOx":   (-2.0,  +2.0),   # often fixed
    "dOy":   (-5.0,  +5.0),
    "dOz":   (-5.0,  +5.0),

    "dDx":   (-10.0, +10.0),
    "dDy":   (-10.0, +10.0),
    "dDz":   (-10.0, +10.0),

    "alpha": (-15, +15),  # degrees
}

REAL_FOLDER = "projections_png_real"
PRED_FOLDER = "projections_png_predicted"

def project_theta_to_bounds(theta, bounds, names):
    theta_clipped = theta.copy()
    for i, name in enumerate(names):
        lo, hi = bounds[name]
        theta_clipped[i] = np.clip(theta_clipped[i], lo, hi)
    return theta_clipped

def print_theta_table(theta, iteration):
    dSx, dSy, dOx, dOy, dOz, dDx, dDy, dDz, alpha = theta

    print("\n" + "=" * 60)
    print(f"Iteration {iteration}")
    print("=" * 60)
    print("Theta offsets (Unity frame):")
    print(f"  Source   : dSx={dSx:8.3f}, dSy={dSy:8.3f}, dSz={0.0:8.3f}  (mm)")
    print(f"  Object   : dOx={dOx:8.3f}, dOy={dOy:8.3f}, dOz={dOz:8.3f}  (mm)")
    print(f"  Detector : dDx={dDx:8.3f}, dDy={dDy:8.3f}, dDz={dDz:8.3f}  (mm)")
    print(f"  Obj rotY : {alpha:8.3f} deg)")
    print("=" * 60)

def make_active_mask(fix_source: bool, fix_detector: bool, fix_object: bool=False, fix_alpha: bool=False):
    """
    Returns boolean mask of length 9 indicating which parameters are optimized.
    """
    mask = np.ones(9, dtype=bool)

    # Source (dSx,dSy)
    if fix_source:
        mask[0] = False
        mask[1] = False

    # Object (dOx,dOy,dOz)
    if fix_object:
        mask[2] = False
        mask[3] = False
        mask[4] = False

    # Detector (dDx,dDy,dDz)
    if fix_detector:
        mask[5] = False
        mask[6] = False
        mask[7] = False

    # alpha
    if fix_alpha:
        mask[8] = False

    return mask

def pack_theta(theta_full: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    return np.asarray(theta_full, dtype=np.float64)[active_mask]

def unpack_theta(theta_free: np.ndarray, theta_full: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(theta_full, dtype=np.float64).copy()
    out[active_mask] = np.asarray(theta_free, dtype=np.float64)
    return out

def apply_theta_to_geometry(theta, src_world, obj_world, det_world, obj_rot_y_deg):
    """
    Apply parameter offsets theta to Unity-world geometry.
    """
    dSx, dSy, dOx, dOy, dOz, dDx, dDy, dDz, alpha = theta

    # Source: shift in detector plane (x,y)
    src_w = src_world + np.array([dSx, dSy, 0.0], dtype=np.float32)
    # Object: full 3D shift
    obj_w = obj_world + np.array([dOx, dOy, dOz], dtype=np.float32)
    # Detector: full 3D shift
    det_w = det_world + np.array([dDx, dDy, dDz], dtype=np.float32)
    # Object rotation (Y axis)
    rot_y = float(obj_rot_y_deg) + alpha

    return src_w, obj_w, det_w, rot_y

def geom12_with_theta(theta, base_src_world, base_obj_world, base_det_world, base_rot_y_deg, 
                      astra_sdd, det_spacing, src_up, src_right):
    theta = np.asarray(theta, dtype=np.float64).reshape(9)
    dSx, dSy = theta[0], theta[1]
    dOx, dOy, dOz = theta[2], theta[3], theta[4]
    dDx, dDy, dDz = theta[5], theta[6], theta[7]
    alpha = theta[8]  # degrees

    src_world = base_src_world + dSx * src_right + dSy * src_up
    obj_world = base_obj_world + np.array([dOx, dOy, dOz], dtype=np.float64)
    det_world = base_det_world + np.array([dDx, dDy, dDz], dtype=np.float64)
    rot_deg = float(base_rot_y_deg + alpha)

    return unity_geom12_from_worldcoords(src_world=src_world, obj_world=obj_world, det_world=det_world, obj_rot_y_deg=rot_deg,
                                         astra_sdd=astra_sdd, det_spacing=det_spacing, src_up=src_up, src_right=src_right)

def project_points_cone_vec(geom12: np.ndarray, bead_xyz: np.ndarray, det_h: int, det_w: int):
    g = np.asarray(geom12, dtype=np.float64).reshape(12)
    S = unpack_xzy(g[0:3])
    D = unpack_xzy(g[3:6])
    u = unpack_xzy(g[6:9])
    v = unpack_xzy(g[9:12])

    n = np.cross(u, v)
    if np.dot(n, n) < 1e-12:
        raise ValueError("Degenerate detector basis (u x v ~ 0).")

    bead_xyz = np.asarray(bead_xyz, dtype=np.float64)
    K = bead_xyz.shape[0]
    out = np.zeros((K, 2), dtype=np.float64)

    for k in range(K):
        X = bead_xyz[k]
        dir_vec = X - S
        denom = np.dot(dir_vec, n)
        if abs(denom) < 1e-12:
            out[k] = np.nan
            continue

        t = np.dot(D - S, n) / denom
        P = S + t * dir_vec
        p_rel = P - D

        M = np.column_stack([u, v])  # 3x2
        coeff, *_ = np.linalg.lstsq(M, p_rel, rcond=None)
        a_u, a_v = coeff[0], coeff[1]

        row = a_u + (det_h - 1) / 2.0
        col = a_v + (det_w - 1) / 2.0
        out[k] = [col, row]  # (x,y)
    return out

def pairwise_dists_xy(X: np.ndarray) -> np.ndarray:
    """
    X: (K,2) array
    returns vector of length K*(K-1)/2 containing pairwise distances, sorted
    """
    K = X.shape[0]
    d = []
    for i in range(K):
        for j in range(i+1, K):
            if np.any(np.isnan(X[i])) or np.any(np.isnan(X[j])):
                continue
            d.append(np.linalg.norm(X[i] - X[j]))
    d = np.asarray(d, dtype=np.float64)
    return np.sort(d)

def match_measured_to_pred(meas: np.ndarray, pred: np.ndarray, area_weight: float = 1e-3):
    K = pred.shape[0]
    best_perm = None
    best_cost = np.inf
    for perm in itertools.permutations(range(K)):
        m = meas[list(perm)]
        diff = (m - pred).copy()
        diff[:, 2] *= area_weight  # weight area if present
        cost = np.sum(diff ** 2)
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
        elif best_perm is None:
            print("WHT?????????????????" + str(diff))
    #print(best_perm)
    return meas[list(best_perm)]

def build_residual(theta, df_wide, bead_xyz, angles_deg, cfg):
    det_h, det_w = cfg["det_h"], cfg["det_w"]
    K = cfg["K"]

    all_r = []
    for i, row in enumerate(df_wide.itertuples(index=False)):
        meas = np.array([[getattr(row, f"x{k+1}"), getattr(row, f"y{k+1}")] for k in range(K)], dtype=np.float64)

        geom12 = geom12_with_theta(
            theta=theta,
            base_src_world=cfg["SRC_WORLD"],
            base_obj_world=cfg["OBJ_WORLD"],
            base_det_world=cfg["DET_WORLD"],
            base_rot_y_deg=float(angles_deg[i]),
            astra_sdd=cfg["ASTRA_SDD"],
            det_spacing=cfg["DET_SPACING"],
            src_up=cfg["SRC_UP"],
            src_right=cfg["SRC_RIGHT"],
        )

        pred = project_points_cone_vec(geom12, bead_xyz, det_h, det_w)

        meas_aligned = match_measured_to_pred(meas, pred)
        diff = pred - meas_aligned  # predicted - measured
        all_r.append(diff.reshape(-1))

    return np.concatenate(all_r, axis=0)


def residual_from_two_dfs(real_df, pred_df, K, area_weight: float = 1e-3, distance_weight: float = 1.0):
    # align by filename
    real_df = real_df.sort_values("image").reset_index(drop=True)
    pred_df = pred_df.sort_values("image").reset_index(drop=True)

    if len(real_df) != len(pred_df):
        raise ValueError(f"real_df has {len(real_df)} rows, pred_df has {len(pred_df)} rows")

    r_list = []
    for i in range(len(real_df)):
        meas = np.array([[real_df.loc[i, f"x{k+1}"], real_df.loc[i, f"y{k+1}"], real_df.loc[i, f"area{k+1}"]] for k in range(K)], dtype=np.float64)
        pred = np.array([[pred_df.loc[i, f"x{k+1}"], pred_df.loc[i, f"y{k+1}"], real_df.loc[i, f"area{k+1}"]] for k in range(K)], dtype=np.float64)

        # --- Point residual (needs correspondence) ---
        meas_aligned = match_measured_to_pred(meas, pred, area_weight=area_weight)
        diff_pts = (pred - meas_aligned).copy()
        diff_pts[:, 2] *= area_weight

        r_list.append(diff_pts.reshape(-1))

        # --- Pairwise distance residual (no correspondence needed) ---
        meas_xy = meas[:, :2]
        pred_xy = pred[:, :2]
        d_meas = pairwise_dists_xy(meas_xy)
        d_pred = pairwise_dists_xy(pred_xy)
        # make same length in case some NaNs reduced pairs
        L = min(d_meas.size, d_pred.size)
        if L > 0:
            diff_d = (d_pred[:L] - d_meas[:L]) * distance_weight
            r_list.append(diff_d.reshape(-1))
    #print(f"Residual length from two dfs: {len(r_list)}")
    return np.concatenate(r_list, axis=0)

def generate_predicted_projections(theta, angles_deg, cfg, out_dir):
    base_rot0 = float(angles_deg[0])
    src_w, obj_w, det_w, rot_y = apply_theta_to_geometry(
            theta=theta,
            src_world=cfg["SRC_WORLD"],
            obj_world=cfg["OBJ_WORLD"],
            det_world=cfg["DET_WORLD"],
            obj_rot_y_deg=base_rot0,
        )

    # For each angle, rotation changes, but src/obj/det translations come from theta
    alpha_deg = float(theta[8])  # if alpha is degrees in your code
    obj_rot_y_degs = np.asarray(angles_deg, dtype=np.float32) + alpha_deg

    fetch_and_save_projections(
        out_dir=out_dir,
        src_world=src_w,
        obj_world=obj_w,
        det_world_base=det_w,
        obj_rot_y_degs=obj_rot_y_degs,
        image_height=cfg["det_h"],
        image_width=cfg["det_w"],
        astra_sdd=cfg["ASTRA_SDD"],
        det_spacing=cfg["DET_SPACING"],
        voxel_size=cfg["VOXEL_SIZE"],
        src_up=cfg["SRC_UP"],
        src_right=cfg["SRC_RIGHT"],
        filename_prefix="proj",
    )


def build_residual_image_based(theta, real_df, angles_deg, cfg, pred_dir):
    # 1) generate predicted projections for current theta
    generate_predicted_projections(theta, angles_deg, cfg, pred_dir)

    # 2) detect beads in predicted projections
    pred_df = build_wide_df_from_folder(
        pred_dir,
        K=cfg["K"],
        min_area=cfg.get("min_area", 10),
        max_area=cfg.get("max_area", 2000),
    )
    if len(pred_df) != len(real_df):
        print(f"FAILED!!!!!!!!!! pred_df has {len(pred_df)} rows, real_df has {len(real_df)} rows")
        return np.empty((0,), dtype=np.float64)

    #print(pred_dir)
    #print(pred_df[["image"] + [f"x{i+1}" for i in range(cfg["K"])] + [f"y{i+1}" for i in range(cfg["K"])]])
    # 3) compare
    return residual_from_two_dfs(real_df, pred_df, cfg["K"])

def numerical_jacobian_image_based(theta, active_mask, real_df, angles_deg, cfg, eps, work_dir):
    r0 = build_residual_image_based(theta, real_df, angles_deg, cfg, os.path.join(work_dir, "pred_base"))
    if len(r0) == 0:
        return None, None
    #print(r0)
    M = r0.size
    active_idx = np.where(active_mask)[0]
    P = active_idx.size
    J = np.zeros((M, P), dtype=np.float64)

    for col, j in enumerate(active_idx):
        t_p = theta.copy()
        t_m = theta.copy()
        t_p[j] += eps[j]; t_m[j] -= eps[j]

        r_p = build_residual_image_based(t_p, real_df, angles_deg, cfg, os.path.join(work_dir, f"pred_p_{j:02d}"))
        r_m = build_residual_image_based(t_m, real_df, angles_deg, cfg, os.path.join(work_dir, f"pred_m_{j:02d}"))
        if  len(r_p) == 0 or len(r_m) == 0:
            continue

        J[:, col] = (r_p - r_m) / (2.0 * eps[j])

    return r0, J


def lm_solve_image_based(real_df, angles_deg, cfg, n_iters=10, lam=1e-2, fix_source=False, fix_detector=False, work_dir="lm_work"):
    os.makedirs(work_dir, exist_ok=True)
    theta = np.zeros(9, dtype=np.float64)
    eps = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    eps = np.ones(9, dtype=np.float64)

    active_mask = make_active_mask(fix_source, fix_detector, fix_object=False, fix_alpha=False)
    # --- stall tracking ---
    dtheta_norm_hist = []
    stall_count = 0

    for it in range(200):
        # ---- PRINT PARAMS ----
        print_theta_table(theta, it)

        # ---- PRINT UNITY + ASTRA GEOM FOR FIRST VIEW ----
        base_rot0 = float(angles_deg[0])
        src_w, obj_w, det_w, rot_y = apply_theta_to_geometry(theta, 
            src_world=cfg["SRC_WORLD"],
            obj_world=cfg["OBJ_WORLD"],
            det_world=cfg["DET_WORLD"], 
            obj_rot_y_deg=base_rot0)
        print_unity_geometry(src_w, obj_w, det_w, rot_y)

        geom12 = unity_geom12_from_worldcoords(
            src_world=src_w,
            obj_world=obj_w,
            det_world=det_w,
            obj_rot_y_deg=rot_y,
            astra_sdd=cfg["ASTRA_SDD"],
            det_spacing=cfg["DET_SPACING"],
            src_up=cfg["SRC_UP"],
            src_right=cfg["SRC_RIGHT"],
        )
        #print_geometry_vector(geom12)

        # ---- RESIDUAL + JACOBIAN (IMAGE-BASED) ----
        r, J = numerical_jacobian_image_based(theta, active_mask, real_df, angles_deg, cfg, eps, work_dir)
        print(f"Residual vector length: {r.size}, Jacobian shape: {J.shape}")

        # ---- GAuss Newton STEP ----
        A = J.T @ J
        g = J.T @ r
        dtheta = -np.linalg.solve(A + lam * np.eye(A.shape[0]), g)
        dtheta_full = np.zeros_like(theta)
        dtheta_full[active_mask] = dtheta
        new_theta = theta + dtheta_full

        # ---- ACCEPT / REJECT ----
        cost = 0.5 * (r @ r)
        r_new = build_residual_image_based(new_theta, real_df, angles_deg, cfg, os.path.join(work_dir, "pred_trial"))
        cost_new = 0.5 * (r_new @ r_new)

        print(f"\niter {it:02d} cost={cost:.6f} -> {cost_new:.6f} |dtheta|={np.linalg.norm(dtheta):.6e}  lambda={lam:.3e}")

        if cost_new < cost:
            theta = new_theta
            lam = max(lam / 3.0, 1e-6)
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
    return theta, f"{np.linalg.norm(dtheta):.6e}", cost_new, it+1

# -----------------------------
# MAIN
# -----------------------------
np.set_printoptions(suppress=True, precision=8)

if __name__ == "__main__":
    IMAGE_W = 255
    IMAGE_H = 255

    ASTRA_SDD = 1
    DET_SPACING = 0.75

    # xraySource orientation (world). Use your real values if different.
    # If your source GameObject has default rotation, these are usually:
    SRC_UP = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    SRC_RIGHT = np.array([0.0, 1.0, 0.0], dtype=np.float32)


    GEOM_SCENARIOS = [
        {
            "name": "G0",
            "SRC_WORLD": np.array([0.0, 45.0, -50.0], dtype=np.float32),
            "DET_WORLD": np.array([0.0, 0.0, 1004.0], dtype=np.float32),
            "real_OBJ_WORLD": np.array([0.0, 30.0, 570.0], dtype=np.float32),
            "unity_OBJ_WORLD": np.array([10.0, 25.0, 589.0], dtype=np.float32),
            "initial_angle_deg": 10.0,
            "fake_delta": np.array([0.0, 0.0, -10.0, 5.0, -19.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float32),
        },
        {
            "name": "G1",
            "SRC_WORLD": np.array([2.0, 45.0, -50.0], dtype=np.float32),
            "DET_WORLD": np.array([0.0, 2.0, 1004.0], dtype=np.float32),
            "real_OBJ_WORLD": np.array([0.0, 30.0, 570.0], dtype=np.float32),
            "unity_OBJ_WORLD": np.array([6.0, 28.0, 585.0], dtype=np.float32),
            "initial_angle_deg": 8.0,
            "fake_delta": np.array([0.0, 0.0, -6.0, 2.0, -15.0, 0.0, 0.0, 0.0, 8.0], dtype=np.float32),
        },
        {
            "name": "G2",
            "SRC_WORLD": np.array([-2.0, 47.0, -50.0], dtype=np.float32),
            "DET_WORLD": np.array([1.0, 0.0, 1006.0], dtype=np.float32),
            "real_OBJ_WORLD": np.array([0.0, 30.0, 570.0], dtype=np.float32),
            "unity_OBJ_WORLD": np.array([14.0, 22.0, 592.0], dtype=np.float32),
            "initial_angle_deg": 12.3,
            "fake_delta": np.array([0.0, 0.0, -14.0,  8.0, -22.0, 0.0, 0.0, 0.0, 12.3], dtype=np.float32),
        },
        {
            "name": "G3",
            "SRC_WORLD": np.array([0.0, 43.0, -48.0], dtype=np.float32),
            "DET_WORLD": np.array([-2.0, 0.0, 1004.0], dtype=np.float32),
            "real_OBJ_WORLD": np.array([0.0, 30.0, 570.0], dtype=np.float32),
            "unity_OBJ_WORLD": np.array([8.0, 24.0, 595.0], dtype=np.float32),
            "initial_angle_deg": 14.1,
            "fake_delta": np.array([0.0, 0.0, -8.0,  6.0, -25.0, 0.0, 0.0, 0.0, 14.1], dtype=np.float32),

        },
        {
            "name": "G4",
            "SRC_WORLD": np.array([1.0, 46.0, -55.0], dtype=np.float32),
            "DET_WORLD": np.array([0.0, -2.0, 1008.0], dtype=np.float32),
            "real_OBJ_WORLD": np.array([0.0, 30.0, 570.0], dtype=np.float32),
            "unity_OBJ_WORLD": np.array([12.0, 27.0, 583.0], dtype=np.float32),
            "initial_angle_deg": 3.3,
            "fake_delta": np.array([0.0, 0.0, -12.0,  3.0, -13.0, 0.0, 0.0, 0.0,  3.3], dtype=np.float32),
        },
    ]


    ANGLE_FACTORS = [3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360]
    #ANGLE_FACTORS = [8]
    BEAD_LIST = range(1, 8)

    MIN_AREA = 10
    MAX_AREA = 6000
    AREA_WEIGHT = 1e-3
    VOXEL_SIZE = 0.1

    # Put outputs in unique places and do not overwrite
    BASE_REAL_DIR = "projections_png_real"
    THETA_TXT = "theta_hat.txt"
    THETA_CSV = "theta_hat.csv"

    # CSV header once (outside loops)THETA_CSV = "theta_hat.csv"
    scenario_names = [sc["name"] for sc in GEOM_SCENARIOS]

    if not os.path.isfile(THETA_CSV):
        with open(THETA_CSV, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)

            header = ["K", "N_ANGLES"]
            for name in scenario_names:
                header += [
                    f"{name}_sum",
                    f"{name}_dtheta",
                    f"{name}_cost",
                    f"{name}_it",
                ]

            writer.writerow(header)


    scenario_results = {}

    for each_k in BEAD_LIST:
        generate_k_bead_phantom(each_k, plot=False)
        for each_angle in ANGLE_FACTORS:
            for sc in GEOM_SCENARIOS:
                scenario_name = sc["name"]

                print("\n" + "#" * 80)
                print(f"Running scenario={scenario_name} N_ANGLES={each_angle}, K={each_k}")
            
                # ---- Fake Flex Ray coordinates ----
                # Default position
                SRC_WORLD = sc["SRC_WORLD"]
                DET_WORLD = sc["DET_WORLD"]
                REAL_OUT_DIR = os.path.join(BASE_REAL_DIR, f"{scenario_name}_K{each_k}_N{each_angle}")
                os.makedirs(REAL_OUT_DIR, exist_ok=True)

                REAL_OBJ_WORLD = sc["real_OBJ_WORLD"]
                start_deg = float(sc["initial_angle_deg"])
                ANGLE_DEGREES_REAL = np.linspace(start_deg, start_deg + 360.0, each_angle, endpoint=False)

            
                print("Generating projections with angles (deg):", ANGLE_DEGREES_REAL)
                fetch_and_save_projections(
                    out_dir="projections_png_real",
                    src_world=SRC_WORLD,
                    obj_world=REAL_OBJ_WORLD,
                    det_world_base=DET_WORLD,
                    obj_rot_y_degs=ANGLE_DEGREES_REAL,
                    image_height=IMAGE_H,
                    image_width=IMAGE_W,
                    astra_sdd=ASTRA_SDD,
                    det_spacing=DET_SPACING,
                    voxel_size=VOXEL_SIZE,
                    src_up=SRC_UP,
                    src_right=SRC_RIGHT,
                    filename_prefix="proj",
                )

                real_proj = build_wide_df_from_folder(REAL_FOLDER, K=each_k, min_area=MIN_AREA, max_area=MAX_AREA)
                print(real_proj)

                # ---- Unity coordinates ----
                # Default position
                UNITY_OBJ_WORLD = sc["unity_OBJ_WORLD"]
                ANGLE_DEGREES_UNITY = np.linspace(0.0, 360.0, each_angle, endpoint=False)

                fake_delta = sc["fake_delta"]  # from dict
                # Ensure 2-decimal accuracy if you want strict formatting:
                fake_delta = np.round(fake_delta.astype(np.float32), 2)

                cfg = {
                    "K": each_k,
                    "det_h": IMAGE_W,
                    "det_w": IMAGE_H,
                    "ASTRA_SDD": ASTRA_SDD,
                    "DET_SPACING": DET_SPACING,
                    "SRC_WORLD": SRC_WORLD,
                    "OBJ_WORLD": UNITY_OBJ_WORLD,
                    "DET_WORLD": DET_WORLD,
                    "VOXEL_SIZE": VOXEL_SIZE,
                    "SRC_UP": SRC_UP,
                    "SRC_RIGHT": SRC_RIGHT,
                    "min_area": MIN_AREA,
                    "max_area": MAX_AREA,
                }
            
                theta_hat, dtheta, cost, it = lm_solve_image_based(real_proj, ANGLE_DEGREES_UNITY, cfg, n_iters=50, lam=1e-2, fix_source=True, fix_detector=True)
                # Diff
                delta_minus_fake = theta_hat.copy()
                delta_minus_fake -= fake_delta
                diff_sum = float(delta_minus_fake.sum())

                scenario_results[sc["name"]] = {
                    "sum": diff_sum,
                    "dtheta": float(dtheta),
                    "cost": float(cost),
                    "it": int(it),
                }


                # -----------------------------------------
                # Text log (append)
                # -----------------------------------------
                with open(THETA_TXT, "a") as ftxt:
                    ftxt.write(f"# scenario={scenario_name} N_ANGLES={each_angle}, K={each_k}\n")
                    ftxt.write("# fake_delta\n")
                    ftxt.write(" ".join(f"{v:.2f}" for v in fake_delta) + "\n")
                    ftxt.write("# theta_hat\n")
                    ftxt.write(" ".join(f"{v:.3f}" for v in theta_hat) + "\n")
                    ftxt.write("# Diff from Expected\n")
                    ftxt.write(" ".join(f"{v:.3f}" for v in delta_minus_fake) + "\n")
                    ftxt.write(f"# sum = {diff_sum:.3f}\n\n")
          
                print("Fake theta:", fake_delta)
                print("Final estimated theta:", theta_hat)
                print("Diff from Expected:", delta_minus_fake)
                print("#" * 80 + "\n")

            with open(THETA_CSV, "a", newline="") as fcsv:
                writer = csv.writer(fcsv)

                row = [each_k, each_angle]
                for name in scenario_names:
                    r = scenario_results[name]
                    row += [
                        f"{r['sum']:.3f}",
                        f"{r['dtheta']:.6f}",
                        f"{r['cost']:.6f}",
                        r["it"],
                    ]

                writer.writerow(row)
