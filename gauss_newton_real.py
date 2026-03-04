import os
import glob
import argparse
from re import A, I
from tkinter import OFF
from bead_detection import build_wide_df_from_folder
import cv2
import csv
import numpy as np
import pandas as pd
import itertools
from pathlib import Path

from phantom_generator import generate_k_bead_phantom
from phantom_projector import fetch_and_save_projections, print_geometry_vector, print_unity_geometry, unity_geom12_from_worldcoords, unpack_xzy

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

BOUNDS = {
    # "dSx":   (-5.0,  +5.0),
    # "dSy":   (-5.0,  +5.0),

    # "dOx":   (-2.0,  +2.0),   # often fixed
    # "dOy":   (-5.0,  +5.0),
    # "dOz":   (-5.0,  +5.0),

    # "dDx":   (-10.0, +10.0),
    # "dDy":   (-10.0, +10.0),
    # "dDz":   (-10.0, +10.0),

    "alpha": (-15, +15),  # degrees
}


def parse_int_list(raw: str):
    if raw is None:
        return None
    parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]

def print_delta_table(delta, iteration):
    dSx, dSy, dOx, dOy, dOz, dDx, dDy, dDz, alpha, offset_x, offset_z = delta

    print("\n" + "=" * 60)
    print(f"Iteration {iteration}")
    print("=" * 60)
    print("delta offsets (Unity frame):")
    print(f"  Source   : dSx={dSx:8.3f}, dSy={dSy:8.3f}, dSz={0.0:8.3f}  (mm)")
    print(f"  Object   : dOx={dOx:8.3f}, dOy={dOy:8.3f}, dOz={dOz:8.3f}  (mm)")
    print(f"  Object offset: offset_x={offset_x:8.3f}, offset_z={offset_z:8.3f} (mm)")
    print(f"  Detector : dDx={dDx:8.3f}, dDy={dDy:8.3f}, dDz={dDz:8.3f}  (mm)")
    print(f"  Obj Stage rotY : {alpha:8.3f} deg")
    print("=" * 60)

def make_active_mask(fix_source: bool, fix_detector: bool, fix_object: bool=False, fix_alpha: bool=False, fix_offset: bool=False):
    """
    Returns boolean mask of length 9 indicating which parameters are optimized.
    """
    mask = np.ones(11, dtype=bool)

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

    if fix_offset:
        mask[9] = False
        mask[10] = False

    return mask

def apply_delta_to_geometry(delta, src_world, obj_world, det_world):
    """
    Apply parameter offsets delta to Unity-world geometry.
    """
    dSx, dSy, dOx, dOy, dOz, dDx, dDy, dDz, alpha, offset_x, offset_z = delta

    # Source: shift in detector plane (x,y)
    src_w = src_world + np.array([dSx, dSy, 0.0], dtype=np.float32)
    # Object: full 3D shift
    obj_w = obj_world + np.array([dOx, dOy, dOz], dtype=np.float32)
    # Detector: full 3D shift
    det_w = det_world + np.array([dDx, dDy, dDz], dtype=np.float32)

    return src_w, obj_w, det_w, alpha, offset_x, offset_z

def match_measured_to_pred(meas: np.ndarray, pred: np.ndarray, area_weight: float = 1e-3):
    K = pred.shape[0]
    best_perm = None
    best_cost = np.inf
    for perm in itertools.permutations(range(K)):
        m = meas[list(perm)]
        diff = (m - pred).copy()
        # diff[:, 2] *= area_weight  # weight area if present
        cost = np.sum(diff ** 2)
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
        elif best_perm is None:
            print("WHT?????????????????" + str(diff))
    return meas[list(best_perm)]


def residual_from_two_dfs(real_df, pred_df, K, area_weight: float = 1e-3, distance_weight: float = 1.0):
    real_df = real_df.sort_values("image").reset_index(drop=True)
    pred_df = pred_df.sort_values("image").reset_index(drop=True)

    if len(real_df) != len(pred_df):
        raise ValueError(f"real_df has {len(real_df)} rows, pred_df has {len(pred_df)} rows")

    r_list = []
    col_names = []

    for i in range(len(real_df)):
        image_id = i + 1  # or use real_df.loc[i, "image"]

        meas = np.array([[real_df.loc[i, f"x{k+1}"], real_df.loc[i, f"y{k+1}"]] for k in range(K)], dtype=np.float64)
        pred = np.array([[pred_df.loc[i, f"x{k+1}"], pred_df.loc[i, f"y{k+1}"]] for k in range(K)], dtype=np.float64)

        # -------------------------------
        # Bead-to-bead residuals
        # -------------------------------
        meas_aligned = match_measured_to_pred(meas, pred, area_weight=area_weight)
        diff_pts = (pred - meas_aligned).copy()

        for k in range(K):
            dx = diff_pts[k, 0]
            dy = diff_pts[k, 1]
            err = np.sqrt(dx**2 + dy**2)

            r_list.append(err)
            col_names.append(f"{image_id}_b{k+1}_b{k+1}")

        # -------------------------------
        # Pairwise residuals
        # -------------------------------
        for a in range(K):
            for b in range(a + 1, K):
                if np.any(np.isnan(pred[a])) or np.any(np.isnan(pred[b])):
                    continue
                if np.any(np.isnan(meas_aligned[a])) or np.any(np.isnan(meas_aligned[b])):
                    continue

                d_pred = np.linalg.norm(pred[a] - pred[b])
                d_meas = np.linalg.norm(meas_aligned[a] - meas_aligned[b])

                diff = (d_pred - d_meas) * distance_weight

                r_list.append(diff)
                col_names.append(f"{image_id}_b{a+1}_b{b+1}")

    r_vec = np.array(r_list, dtype=np.float64)

    return r_vec, col_names

def generate_predicted_projections(delta, angles_deg, cfg, out_dir):
    src_w, obj_w, det_w, alpha, offset_x, offset_z = apply_delta_to_geometry(
        delta=delta,
        src_world=cfg["SRC_WORLD"],
        obj_world=cfg["OBJ_WORLD"],
        det_world=cfg["DET_WORLD"],
    )
    fetch_and_save_projections(
        out_dir=out_dir,
        src_world=src_w,
        obj_world=obj_w,
        det_world_base=det_w,
        alpha=alpha,
        angles_deg=angles_deg,
        offset_x=offset_x,
        offset_z=offset_z,
        image_height=cfg["det_h"],
        image_width=cfg["det_w"],
        astra_scaling=cfg["astra_scaling"],
        det_spacing=cfg["DET_SPACING"],
        voxel_size=cfg["VOXEL_SIZE"],
        src_up=cfg["SRC_UP"],
        src_right=cfg["SRC_RIGHT"],
        filename_prefix="proj",
        #phantom_name=HERE/f"phantoms/cuboid_phantom_{cfg['K']}.npy"
        phantom_name=HERE/f"phantoms/scan2_160x240x498.npy"
        
    )

def build_residual_image_based(delta, real_df, angles_deg, cfg, pred_dir):
    # 1) generate predicted projections for current delta
    generate_predicted_projections(delta, angles_deg, cfg, pred_dir)

    # 2) detect beads in predicted projections
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

    #print(pred_dir)
    #print(pred_df[["image"] + [f"x{i+1}" for i in range(cfg["K"])] + [f"y{i+1}" for i in range(cfg["K"])]])
    # 3) compare
    return residual_from_two_dfs(real_df, pred_df, cfg["K"])

def numerical_jacobian_image_based(delta, active_mask, real_df, angles_deg, cfg, eps, work_dir):
    r0, cols = build_residual_image_based(delta, real_df, angles_deg, cfg, work_dir / "pred_base")

    if len(r0) == 0:
        return None, None
    M = r0.size
    active_idx = np.where(active_mask)[0]
    P = active_idx.size
    J = np.zeros((M, P), dtype=np.float64)

    for col, j in enumerate(active_idx):
        t_p = delta.copy()
        t_m = delta.copy()
        t_p[j] += eps[j]; t_m[j] -= eps[j]

        r_p, _ = build_residual_image_based(t_p, real_df, angles_deg, cfg, work_dir / f"pred_p_{j:02d}")
        r_m, _ = build_residual_image_based(t_m, real_df, angles_deg, cfg, work_dir / f"pred_m_{j:02d}")
        if  len(r_p) == 0 or len(r_m) == 0:
            continue

        J[:, col] = (r_p - r_m) / (2.0 * eps[j])
    return r0, J, cols

def lm_solve_image_based(real_df, angles_deg, cfg, n_iters=10, lam=1e-2, fix_source=False, fix_detector=False, fix_object=False, fix_offset=False, work_dir="lm_work"):
    os.makedirs(work_dir, exist_ok=True)
    delta = np.zeros(11, dtype=np.float64)
    eps = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01], dtype=np.float64)
    # eps = np.ones(9, dtype=np.float64)

    active_mask = make_active_mask(fix_source, fix_detector, fix_object, fix_alpha=False, fix_offset=fix_offset)
    # --- stall tracking ---
    ddelta_norm_hist = []
    stall_count = 0

    df_r0 = None
    for it in range(200):
        # ---- PRINT PARAMS ----
        print_delta_table(delta, it)

        # ---- PRINT UNITY + ASTRA GEOM FOR FIRST VIEW ----
        src_w, obj_w, det_w, alpha, offset_x, offset_z = apply_delta_to_geometry(delta, 
            src_world=cfg["SRC_WORLD"],
            obj_world=cfg["OBJ_WORLD"],
            det_world=cfg["DET_WORLD"],
        )
        print_unity_geometry(src_w, obj_w, det_w, angles_deg[0])

        # ---- RESIDUAL + JACOBIAN (IMAGE-BASED) ----
        r, J, cols = numerical_jacobian_image_based(delta, active_mask, real_df, angles_deg, cfg, eps, work_dir)
        r1 = np.asarray(r, dtype=np.float64).reshape(-1)

        print(f"Residual vector length: {r1.size}, Jacobian shape: {J.shape}")

        # ---- Gauss Newton STEP ----
        A = J.T @ J
        g = J.T @ r1
        ddelta = -np.linalg.solve(A + lam * np.eye(A.shape[0]), g)
        ddelta_full = np.zeros_like(delta)
        ddelta_full[active_mask] = ddelta
        new_delta = delta + ddelta_full
        
        cost = 0.5 * float(r1 @ r1)

        r_new, cols_new = build_residual_image_based(new_delta, real_df, angles_deg, cfg, os.path.join(work_dir, "pred_trial"))
        
        if cols_new != cols:
            raise ValueError("Column names/order mismatch between base and trial residuals")

        r2 = np.asarray(r_new, dtype=np.float64).reshape(-1)
        cost_new = 0.5 * float(r2 @ r2)
        df_iter = pd.DataFrame(
            [r1, r2],
            columns=cols
        )
        df_iter.insert(0, "iter", it)
        df_iter.insert(1, "state", ["base", "trial"])
        df_iter.insert(2, "cost", [cost, cost_new])
        if df_r0 is None:
            df_r0 = pd.DataFrame(columns=["iter", "state", "cost"] + cols)
        df_r0 = pd.concat([df_r0, df_iter], ignore_index=True)

        print(f"\niter {it:02d} cost={cost:.6f} -> {cost_new:.6f} |ddelta|={np.linalg.norm(ddelta):.6e}  lambda={lam:.3e}")

        if cost_new < cost:
            delta = new_delta
            lam = max(lam / 3.0, 1e-6)
        else:
            lam = min(lam * 5.0, 1e6)


        if np.linalg.norm(ddelta) < 1e-6:
            print("Converged.")
            break
        ddelta_norm_hist.append(ddelta)
        if len(ddelta_norm_hist) >= 5:
            recent_norms = [np.linalg.norm(dn) for dn in ddelta_norm_hist[-5:]]
            if max(recent_norms) - min(recent_norms) < 1e-8:
                stall_count += 1
                if stall_count >= 3:
                    print("Stalled.")
                    break
            else:
                stall_count = 0

    print("\nEstimated delta:")
    for name, v in zip(PARAM_NAMES, delta):
        if name == "alpha":
            print(f"{name:>5s}: {v:+.6e} deg)")
        else:
            print(f"{name:>5s}: {v:+.6f}")
    print("\nFinal cost table:")
    df_r0.to_csv(os.path.join(work_dir, "residual_history.csv"), index=False)
    return delta, f"{np.linalg.norm(ddelta):.6e}", cost_new, it+1


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gauss-Newton calibration runner")
    parser.add_argument(
        "-a"
        "--angles",
        "--angle-factors",
        dest="angles",
        default=None,
        help="Comma/space-separated list of N_ANGLE values. Example: '8,12,24'",
    )
    parser.add_argument(
        "-s"
        "--s",
        "--scenerio",
        dest="scenerio",
        default=None,
        help="Comma/space-separated list of K values. Example: '3,4'",
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
            'src':np.array([ 4.999512, 29.994888,  0.      ], dtype=np.float32),
            'det':np.array([ -25.31836 ,   18.676949, 1059.      ], dtype=np.float32),
            'obj':np.array([  0.540527, 20 , 600.      ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan3',
            'src':np.array([ 4.999512, 29.99237 ,  0.      ], dtype=np.float32),
            'det':np.array([ -25.31836 ,   18.666954, 1059.      ], dtype=np.float32),
            'obj':np.array([  0.540527, 20 , 799.9995  ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan4',
            'src':np.array([-10.000488,  29.997368,   0.      ], dtype=np.float32),
            'det':np.array([ -20.002441,   33.6557  , 1059.      ], dtype=np.float32),
            'obj':np.array([  0.540527, 20 , 799.9995  ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan5',
            'src':np.array([-10.000488,  29.997368,   0.      ], dtype=np.float32),
            'det':np.array([-20.002441,  33.6557  , 959.      ], dtype=np.float32),
            'obj':np.array([  0.540527, 20 , 600.      ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan6',
            'src':np.array([ 0.000488, 29.997368,  0.      ], dtype=np.float32),
            'det':np.array([ -6.002441,  33.6557  , 959.      ], dtype=np.float32),
            'obj':np.array([  0.540527, 20 , 600.      ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan7',
            'src':np.array([ 4.999512, 29.994888,  0.      ], dtype=np.float32),
            'det':np.array([ -25.31836 ,   18.676949, 1059.      ], dtype=np.float32),
            'obj':np.array([  0.540527, 20  , 600.      ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan8',
            'src':np.array([ 4.999512, 29.99237 ,  0.      ], dtype=np.float32),
            'det':np.array([ -25.31836 ,   18.666954, 1059.      ], dtype=np.float32),
            'obj':np.array([  0.540527, 20 , 799.9995  ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan9',
            'src':np.array([-10.000488,  29.997368,   0.      ], dtype=np.float32),
            'det':np.array([ -20.002441,   33.6557  , 1059.      ], dtype=np.float32),
            'obj':np.array([  0.540527, 20 , 799.9995  ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan10',
            'src': np.array([-10.000488,  29.997368,   0.      ], dtype=np.float32),
            'det':np.array([-20.002441,  33.6557  , 959.      ], dtype=np.float32),
            'obj':np.array([  0.540527, 20 , 600.      ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
        {
            'name': 'Scan11',
            'src':np.array([ 0.000488, 29.997368,  0.      ], dtype=np.float32),
            'det':np.array([ -6.002441,  33.6557  , 959.      ], dtype=np.float32),
            'obj':np.array([  0.540527, 20 , 600.      ], dtype=np.float32),
            'initial_angle_deg': -4.569823,
            'projections': 360,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
        },
    ]
    
    astra_scaling = 1

    SRC_UP = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    SRC_RIGHT = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    BEAD_COUNT = K = 5
    ANGLE_FACTORS = [3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360]

    MIN_AREA = 200
    MAX_AREA = 6000
    AREA_WEIGHT = 1e-3
    VOXEL_SIZE = 0.1

    PHANTOM_PATH = HERE / "phantoms/scan2_160x240x498.npy"
    BASE_REAL_DIR = HERE / "real_scans/2026-02-19_Beads_phantom"

    used_projections = [360]

    if args.angles is not None:
        used_projections = parse_int_list(args.angles)
        if used_projections is None:
            print("Invalid angles argument, using default.")
            used_projections = 360

    if args.scenerio is not None:
        GEOM_SCENARIOS = [sc for sc in GEOM_SCENARIOS if sc["name"] in parse_int_list(args.scenerio)]

    for each_no_projections in used_projections:
        for sc in GEOM_SCENARIOS:
            scenario_name = sc["name"]
            projections = sc["projections"]

            print("\n" + "#" * 80)
            print(f"Running scenario={scenario_name} Projections={projections} Used={each_no_projections}, K={K}")

            indices = np.linspace(0, projections - 1, each_no_projections, dtype=int)
            real_out_dir = BASE_REAL_DIR / f"{scenario_name}" / f"out_line_integrals"

            # Get bead positiond for real projections
            real_proj = build_wide_df_from_folder(real_out_dir, K=K, min_area=MIN_AREA, max_area=MAX_AREA, file_type=".tif", tolerance=130, indices=indices, box_images=True)

            # ---- Unity coordinates ----
            # Default position
            start_deg = float(sc["initial_angle_deg"])
            projection_angles = np.linspace(start_deg, start_deg + 360.0, each_no_projections, endpoint=False)

            cfg = {
                "K": K,
                "det_h": sc["image_height"],
                "det_w": sc["image_width"],
                "astra_scaling": astra_scaling,
                "DET_SPACING": sc["det_spacing"],
                "SRC_WORLD": sc["src"],
                "OBJ_WORLD": sc["obj"],
                "DET_WORLD": sc["det"],
                "VOXEL_SIZE": VOXEL_SIZE,
                "SRC_UP": SRC_UP,
                "SRC_RIGHT": SRC_RIGHT,
                "min_area": MIN_AREA,
                "max_area": MAX_AREA,
                "box_images": True,
            }
            os.makedirs(HERE / f"lm_work_real", exist_ok=True)
            delta_hat, ddelta, cost, it = lm_solve_image_based(real_proj, projection_angles, cfg, n_iters=50, lam=1e-2, fix_source=True, fix_detector=True, fix_object=False, fix_offset=False, work_dir = HERE / f"fake_projections/lm_work_real/{each_no_projections}" / f"{scenario_name}")
            # Diff
            # delta_minus_fake = delta_hat.copy()
            # delta_minus_fake -= fake_delta
            # diff_sum = float(delta_minus_fake.sum())

            # -----------------------------------------
            # Text log (append)
            # -----------------------------------------
            os.makedirs(HERE / f"delta_log", exist_ok=True)
            delta_TXT = HERE / f"delta_log/delta_hat_{scenario_name}_{each_no_projections}.txt"
            with open(delta_TXT, "a") as ftxt:
                ftxt.write(f"# scenario={scenario_name} N_ANGLES={each_no_projections}, K={K}\n")
                ftxt.write(f"# final_cost={cost:.6f}, iterations={it}, final_ddelta_norm={ddelta}\n")
                #ftxt.write("# fake_delta\n")
                #ftxt.write(" ".join(f"{v:.2f}" for v in fake_delta) + "\n")
                ftxt.write("# delta_hat\n")
                ftxt.write(" ".join(f"{v:.3f}" for v in delta_hat) + "\n")
                #ftxt.write("# Diff from Expected\n")
                #ftxt.write(" ".join(f"{v:.3f}" for v in delta_minus_fake) + "\n")
                #ftxt.write(f"# sum = {diff_sum:.3f}\n\n")
            # print("Fake delta:", fake_delta)
            print("Final estimated delta:", delta_hat)
            # print("Diff from Expected:", delta_minus_fake)
            print("#" * 80 + "\n")
