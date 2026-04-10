import os
import argparse
import csv
from re import A, I
from tkinter import OFF
from bead_detection import build_wide_df_from_folder
import numpy as np
import pandas as pd
import itertools
from pathlib import Path

from phantom_generator import generate_k_bead_phantom
from phantom_projector import fetch_and_save_projections, print_unity_geometry, unity_geom12_from_world_coords, unpack_xzy

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
    print(f"  Source   : dSx={dSx:8.3f}, dSy={dSy:8.3f}, dSz={0.0:8.3f}  (mm)")
    print(f"  Object   : dOx={dOx:8.3f}, dOy={dOy:8.3f}, dOz={dOz:8.3f}  (mm)")
    print(f"  Object offset: offset_x={offset_x:8.3f}, offset_z={offset_z:8.3f}  (mm)")
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

def apply_theta_to_geometry(theta, src_world, obj_world, det_world):
    """
    Apply parameter offsets theta to Unity-world geometry.
    """
    dSx, dSy, dOx, dOy, dOz, dDx, dDy, dDz, alpha, offset_x, offset_z = theta

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

            r_list.append(dx)
            col_names.append(f"{image_id}_b{k+1}_x")
            r_list.append(dy)
            col_names.append(f"{image_id}_b{k+1}_y")

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

def build_residual_image_based(theta, real_df, angles_deg, cfg, pred_dir, debug=True):
    # 1) generate predicted projections for current theta
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
        initial_calibration=cfg['initial_calibration'],
        astra_scaling=cfg["astra_scaling"],
        det_spacing=cfg["DET_SPACING"],
        voxel_size=cfg["VOXEL_SIZE"],
        det_col=cfg["DET_COL"],
        det_row=cfg["DET_ROW"],
        filename_prefix="proj",
        phantom_name=HERE/f"phantoms/cuboid_phantom_{cfg['K']}_{cfg['cuboid_name']}.npy",
        debug=debug
    )

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

def numerical_jacobian_image_based(theta, active_mask, real_df, angles_deg, cfg, eps, work_dir):
    r0, cols = build_residual_image_based(theta, real_df, angles_deg, cfg, work_dir / "pred_base", debug=True)

    if len(r0) == 0:
        return None, None
    M = r0.size
    active_idx = np.where(active_mask)[0]
    P = active_idx.size
    J = np.zeros((M, P), dtype=np.float64)

    for col, j in enumerate(active_idx):
        t_p = theta.copy()
        t_m = theta.copy()
        t_p[j] += eps[j]; t_m[j] -= eps[j]

        r_p, _ = build_residual_image_based(t_p, real_df, angles_deg, cfg, work_dir / f"pred_p_{j:02d}", debug=False)
        r_m, _ = build_residual_image_based(t_m, real_df, angles_deg, cfg, work_dir / f"pred_m_{j:02d}", debug=False)
        if  len(r_p) == 0 or len(r_m) == 0:
            continue

        J[:, col] = (r_p - r_m) / (2.0 * eps[j])
    return r0, J, cols

def lm_solve_image_based(real_df, angles_deg, cfg, n_iters=10, lam=1e-2, fix_source=False, fix_detector=False, fix_object=False, fix_offset=False, work_dir="lm_work"):
    os.makedirs(work_dir, exist_ok=True)
    theta = np.zeros(11, dtype=np.float64)
    eps = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01], dtype=np.float64)
    # eps = np.ones(9, dtype=np.float64)

    active_mask = make_active_mask(fix_source, fix_detector, fix_object, fix_alpha=False, fix_offset=fix_offset)
    # --- stall tracking ---
    dtheta_norm_hist = []
    stall_count = 0

    df_r0 = None
    for it in range(200):
        # ---- PRINT PARAMS ----
        print_theta_table(theta, it)

        # ---- PRINT UNITY + ASTRA GEOM FOR FIRST VIEW ----
        src_w, obj_w, det_w, _, _, _ = apply_theta_to_geometry(theta, 
            src_world=cfg["SRC_WORLD"],
            obj_world=cfg["OBJ_WORLD"],
            det_world=cfg["DET_WORLD"],
        )
        print_unity_geometry(src_w, obj_w, det_w, angles_deg[0])

        # ---- RESIDUAL + JACOBIAN (IMAGE-BASED) ----
        r, J, cols = numerical_jacobian_image_based(theta, active_mask, real_df, angles_deg, cfg, eps, work_dir)
        r1 = np.asarray(r, dtype=np.float64).reshape(-1)

        print(f"Residual vector length: {r1.size}, Jacobian shape: {J.shape}")

        # ---- Gauss Newton STEP ----
        A = J.T @ J
        g = J.T @ r1
        dtheta = -np.linalg.solve(A + lam * np.eye(A.shape[0]), g)
        dtheta_full = np.zeros_like(theta)
        dtheta_full[active_mask] = dtheta
        new_theta = theta + dtheta_full
        
        cost = 0.5 * float(r1 @ r1)

        r_new, cols_new = build_residual_image_based(new_theta, real_df, angles_deg, cfg, os.path.join(work_dir, "pred_trial"))
        
        if cols_new != cols:
            raise ValueError("Column names/order mismatch between base and trial residuals")

        r2 = np.asarray(r_new, dtype=np.float64).reshape(-1)
        cost_new = 0.5 * float(r2 @ r2)
        df_iter = pd.DataFrame([r1, r2], columns=cols)
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
            if lam > 0:
                lam = max(lam / 3.0, 1e-6)
        else:
            if lam > 0:
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
    return theta, f"{np.linalg.norm(dtheta):.6e}", cost_new, it+1


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
        "-k"
        "--k",
        "--bead-list",
        dest="k",
        default=None,
        help="Comma/space-separated list of K values. Example: '3,4'",
    )
    parser.add_argument(
        "-s"
        "--s",
        "--scenario",
        dest="scenario",
        default=None,
        help="Comma/space-separated list of K values. Example: 'Scan3,Scan4'",
    )
    parser.add_argument(
        "-c",
        "--cuboid",
        "--cuboid-size",
        dest="cuboid",
        default=None,
        help="Name of a single cuboid size to run. Example: 'compact', 'medium'",
    )
    parser.add_argument(
        "-l",
        "--lambda",
        "--lambda-name",
        dest="lam",
        default=None,
        help="Lambda variant to run. Example: 'GN', 'LM_low', 'LM_high'",
    )
    args = parser.parse_args()

    astra_scaling = 1

    DET_ROW = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    DET_COL = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    GEOM_SCENARIOS = [
        {
            "name": "G0",
            "SRC_WORLD": np.array([0.0, 45.0, 0.0], dtype=np.float32),
            "DET_WORLD": np.array([0.0, 0.0, 674.0], dtype=np.float32),
            "real_OBJ_WORLD": np.array([0.0, 20.0, 570.0], dtype=np.float32),
            "unity_OBJ_WORLD": np.array([10.0, 15.0, 589.0], dtype=np.float32),
            "initial_angle_deg": 10.0,
            "fake_theta": np.array([0.0, 0.0, -10.0, 5.0, -19.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0], dtype=np.float32),
        },
        {
            "name": "G1",
            "SRC_WORLD": np.array([2.0, 45.0, 0.0], dtype=np.float32),
            "DET_WORLD": np.array([0.0, 2.0, 704.0], dtype=np.float32),
            "real_OBJ_WORLD": np.array([0.0, 20.0, 570.0], dtype=np.float32),
            "unity_OBJ_WORLD": np.array([6.0, 18.0, 585.0], dtype=np.float32),
            "initial_angle_deg": 8.0,
            "fake_theta": np.array([0.0, 0.0, -6.0, 2.0, -15.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0], dtype=np.float32),
        },
        {
            "name": "G2",
            "SRC_WORLD": np.array([-2.0, 47.0, 0.0], dtype=np.float32),
            "DET_WORLD": np.array([1.0, 47.0, 696.0], dtype=np.float32),
            "real_OBJ_WORLD": np.array([0.0, 40.0, 570.0], dtype=np.float32),
            "unity_OBJ_WORLD": np.array([14.0, 32.0, 592.0], dtype=np.float32),
            "initial_angle_deg": 12.3,
            "fake_theta": np.array([0.0, 0.0, -14.0,  8.0, -22.0, 0.0, 0.0, 0.0, 12.3, 0.0, 0.0], dtype=np.float32),
        },
        {
            "name": "G3",
            "SRC_WORLD": np.array([0.0, 43.0, 0.0], dtype=np.float32),
            "DET_WORLD": np.array([-2.0, 0.0, 704.0], dtype=np.float32),
            "real_OBJ_WORLD": np.array([0.0, 20.0, 570.0], dtype=np.float32),
            "unity_OBJ_WORLD": np.array([8.0, 14.0, 595.0], dtype=np.float32),
            "initial_angle_deg": 14.1,
            "fake_theta": np.array([0.0, 0.0, -8.0,  6.0, -25.0, 0.0, 0.0, 0.0, 14.1, 0.0, 0.0], dtype=np.float32),

        },
        {
            "name": "G4",
            "SRC_WORLD": np.array([1.0, 46.0, 0.0], dtype=np.float32),
            "DET_WORLD": np.array([0.0, -2.0, 808.0], dtype=np.float32),
            "real_OBJ_WORLD": np.array([0.0, 20.0, 570.0], dtype=np.float32),
            "unity_OBJ_WORLD": np.array([12.0, 17.0, 583.0], dtype=np.float32),
            "initial_angle_deg": 3.3,
            "fake_theta": np.array([0.0, 0.0, -12.0,  3.0, -13.0, 0.0, 0.0, 0.0,  3.3, 0.0, 0.0], dtype=np.float32),
        },
    ]

    BEAD_LIST = list(range(1, 8))
    ANGLE_FACTORS = [3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360]

    LAMBDA_VALUES = [
        {"name": "GN",      "lam": 0.0},
        {"name": "LM_low",  "lam": 1e-4},
        {"name": "LM_normal",  "lam": 1e-2},
        {"name": "LM_high", "lam": 1.0},
    ]

    CUBOID_SIZES = [
        # Compact — beads close together, minimal spread
        {"name": "compact",    "width": 10.0, "breadth": 10.0, "height": 20.0},
        # Small
        {"name": "small",      "width": 20.0, "breadth": 20.0, "height": 40.0},
        # Normal — your current default
        {"name": "normal",     "width": 20.0, "breadth": 40.0, "height": 60.0},
        # Square
        {"name": "square",     "width": 30.0, "breadth": 30.0, "height": 30.0},
        # Tall — good vertical spread, poor lateral
        {"name": "tall",       "width": 10.0, "breadth": 10.0, "height": 80.0},
        # Wide — good lateral spread, poor vertical
        {"name": "wide",       "width": 80.0, "breadth": 80.0, "height": 20.0},
        # Coplanar — all beads at same height, tests degeneracy
        {"name": "coplanar",   "width": 40.0, "breadth": 40.0, "height": 5.0},
    ]

    cli_angles = parse_int_list(args.angles)
    cli_beads = parse_int_list(args.k)
    if cli_angles is not None:
        ANGLE_FACTORS = cli_angles
    if cli_beads is not None:
        BEAD_LIST = cli_beads
    
    if args.scenario is not None:
        GEOM_SCENARIOS = [sc for sc in GEOM_SCENARIOS if sc["name"] == args.scenario]

    if args.cuboid is not None:
        CUBOID_SIZES = [cs for cs in CUBOID_SIZES if cs["name"] == args.cuboid]

    if args.lam is not None:
        LAMBDA_VALUES = [lv for lv in LAMBDA_VALUES if lv["name"] == args.lam]

    MIN_AREA = 200
    MAX_AREA = 6000
    AREA_WEIGHT = 1e-3
    VOXEL_SIZE = 0.1

    # Put outputs in unique places and do not overwrite
    BASE_REAL_DIR = HERE / f"simulated/real_scans"

    initial_calibration = np.array([
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32), 
        np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ])

    scenario_results = {}

    for each_lambda in LAMBDA_VALUES:
        for each_cuboid in CUBOID_SIZES:
            cuboid_name = each_cuboid["name"]
            for each_k in BEAD_LIST:
                generate_k_bead_phantom(
                    each_k, plot=False,
                    width=each_cuboid["width"],
                    breadth=each_cuboid["breadth"],
                    height=each_cuboid["height"],
                    name=cuboid_name,
                )
                for each_angle in ANGLE_FACTORS:
                    for sc in GEOM_SCENARIOS:
                        scenario_name = sc["name"]

                        print("\n" + "#" * 80)
                        print(f"Running cuboid={cuboid_name} scenario={scenario_name} N_ANGLES={each_angle}, K={each_k}")

                        # ---- Fake Flex Ray coordinates ----
                        # Default position
                        SRC_WORLD = sc["SRC_WORLD"]
                        DET_WORLD = sc["DET_WORLD"]
                        real_out_dir = BASE_REAL_DIR / cuboid_name / each_lambda['name'] / f"K{each_k}_N{each_angle}" / f"{scenario_name}"
                        real_out_dir.mkdir(parents=True, exist_ok=True)

                        REAL_OBJ_WORLD = sc["real_OBJ_WORLD"]
                        start_deg = float(sc["initial_angle_deg"])
                        ANGLE_DEGREES_REAL = np.linspace(0.0, 360.0, each_angle, endpoint=False)

                        # print("Generating projections with angles (deg):", ANGLE_DEGREES_REAL)
                        fetch_and_save_projections(
                            out_dir=real_out_dir,
                            src_world=SRC_WORLD,
                            obj_world=REAL_OBJ_WORLD,
                            det_world_base=DET_WORLD,
                            alpha=start_deg,
                            angles_deg=ANGLE_DEGREES_REAL,
                            offset_x=0.0,
                            offset_z=0.0,
                            image_height=760,
                            image_width=956,
                            initial_calibration=initial_calibration,
                            astra_scaling=astra_scaling,
                            det_spacing=0.149600,
                            voxel_size=VOXEL_SIZE,
                            det_col=DET_COL,
                            det_row=DET_ROW,
                            filename_prefix="proj",
                            phantom_name=HERE/f"phantoms/cuboid_phantom_{each_k}_{cuboid_name}.npy"
                            # phantom_name=HERE/f"phantoms/recon_cropped_scan_1.npy"
                        )

                        real_proj = build_wide_df_from_folder(real_out_dir, K=each_k, min_area=MIN_AREA, max_area=MAX_AREA)
                        print(real_proj)

                        # ---- Unity coordinates ----
                        # Default position
                        UNITY_OBJ_WORLD = sc["unity_OBJ_WORLD"]
                        ANGLE_DEGREES_UNITY = np.linspace(0.0, 360.0, each_angle, endpoint=False)

                        fake_theta = sc["fake_theta"]  # from dict
                        # Ensure 2-decimal accuracy if you want strict formatting:
                        fake_theta = np.round(fake_theta.astype(np.float32), 2)

                        cfg = {
                            "K": each_k,
                            "cuboid_name": cuboid_name,
                            "det_h": 760,
                            "det_w": 956,
                            "astra_scaling": astra_scaling,
                            "DET_SPACING": 0.149600,
                            "SRC_WORLD": SRC_WORLD,
                            "OBJ_WORLD": UNITY_OBJ_WORLD,
                            "DET_WORLD": DET_WORLD,
                            "VOXEL_SIZE": VOXEL_SIZE,
                            "DET_COL": DET_COL,
                            "DET_ROW": DET_ROW,
                            "min_area": MIN_AREA,
                            "max_area": MAX_AREA,
                            "initial_calibration": initial_calibration,
                            "box_images": True,
                        }
                        lambda_name = each_lambda["name"]
                        theta_hat, dtheta, cost, it = lm_solve_image_based(real_proj, ANGLE_DEGREES_UNITY, cfg, n_iters=50, lam=each_lambda["lam"], fix_source=True, fix_detector=True, fix_object=False, fix_offset=False, work_dir = HERE / f"simulated/trial3/{lambda_name}/{cuboid_name}/{each_k}_{each_angle}" / f"{scenario_name}")
                        # Diff
                        theta_minus_fake = theta_hat.copy()
                        theta_minus_fake -= fake_theta
                        diff_sum = float(abs(theta_minus_fake).sum())

                        # scenario_results[sc["name"]] = {
                        #     "sum": diff_sum,
                        #     "dtheta": float(dtheta),
                        #     "cost": float(cost),
                        #     "it": int(it),
                        # }

                        # -----------------------------------------
                        # Text log (append)
                        # -----------------------------------------
                        os.makedirs(HERE / f"simulated/theta_log_lambda", exist_ok=True)
                        THETA_TXT = HERE / f"simulated/theta_log_lambda/theta_hat_{lambda_name}_{cuboid_name}_{each_k}_{each_angle}.txt"
                        with open(THETA_TXT, "a") as ftxt:
                            ftxt.write(f"# lambda={lambda_name} cuboid={cuboid_name} scenario={scenario_name} N_ANGLES={each_angle}, K={each_k}\n")
                            ftxt.write("# fake_theta\n")
                            ftxt.write(" ".join(f"{v:.2f}" for v in fake_theta) + "\n")
                            ftxt.write("# theta_hat\n")
                            ftxt.write(" ".join(f"{v:.3f}" for v in theta_hat) + "\n")
                            ftxt.write("# Diff from Expected\n")
                            ftxt.write(" ".join(f"{v:.3f}" for v in theta_minus_fake) + "\n")
                            ftxt.write(f"# sum = {diff_sum:.3f}\n\n")
                        print("Fake theta:", fake_theta)
                        print("Final estimated theta:", theta_hat)
                        print("Diff from Expected:", theta_minus_fake)
                        print("#" * 80 + "\n")

                # with open(HERE / f"simulated/, "a", newline="") as fcsv:
                #     writer = csv.writer(fcsv)

                #     row = [each_k, each_angle]
                #     for name in scenario_names:
                #         r = scenario_results[name]
                #         row += [
                #             f"{r['sum']:.3f}",
                #             f"{r['dtheta']:.6f}",
                #             f"{r['cost']:.6f}",
                #             r["it"],
                #         ]

                #     writer.writerow(row)
