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

# Fixed ground-truth offsets — identical across all geometry scenarios.
# dOx=-10, dOy=5, dOz=-19  →  real_OBJ = unity_OBJ + [dOx, dOy, dOz]
# dDx=2,  dDy=-3, dDz=0    →  real_DET = unity_DET + [dDx, dDy, 0]
#   dDz is fixed to 0: SDD (detector z) is assumed known for each scenario.
# alpha=10 deg — initial rotation of the object stage in every scenario
FAKE_THETA = np.array(
    [0.0, 0.0, -10.0, 5.0, -19.0, 8.0, -7.0, 0.0, 10.0, 0.0, 0.0],
    dtype=np.float32,
)

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

def make_active_mask(fix_source: bool, fix_detector: bool, fix_object: bool=False, fix_alpha: bool=False, fix_offset: bool=False, fix_det_z: bool=False):
    mask = np.ones(11, dtype=bool)

    if fix_source:
        mask[0] = False
        mask[1] = False

    if fix_object:
        mask[2] = False
        mask[3] = False
        mask[4] = False

    if fix_detector:
        mask[5] = False
        mask[6] = False
        mask[7] = False

    if fix_det_z:
        mask[7] = False  # dDz — SDD direction, highly degenerate with dOz

    if fix_alpha:
        mask[8] = False

    if fix_offset:
        mask[9] = False
        mask[10] = False

    return mask

def apply_theta_to_geometry(theta, src_world, obj_world, det_world):
    dSx, dSy, dOx, dOy, dOz, dDx, dDy, dDz, alpha, offset_x, offset_z = theta

    src_w = src_world + np.array([dSx, dSy, 0.0], dtype=np.float32)
    obj_w = obj_world + np.array([dOx, dOy, dOz], dtype=np.float32)
    det_w = det_world + np.array([dDx, dDy, dDz], dtype=np.float32)

    return src_w, obj_w, det_w, alpha, offset_x, offset_z

def match_measured_to_pred(meas: np.ndarray, pred: np.ndarray, area_weight: float = 1e-3):
    K = pred.shape[0]
    best_perm = None
    best_cost = np.inf
    for perm in itertools.permutations(range(K)):
        m = meas[list(perm)]
        diff = (m - pred).copy()
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
        image_id = i + 1

        meas = np.array([[real_df.loc[i, f"x{k+1}"], real_df.loc[i, f"y{k+1}"]] for k in range(K)], dtype=np.float64)
        pred = np.array([[pred_df.loc[i, f"x{k+1}"], pred_df.loc[i, f"y{k+1}"]] for k in range(K)], dtype=np.float64)

        meas_aligned = meas
        diff_pts = (pred - meas_aligned).copy()

        for k in range(K):
            dx = diff_pts[k, 0]
            dy = diff_pts[k, 1]

            r_list.append(dx)
            col_names.append(f"{image_id}_b{k+1}_x")
            r_list.append(dy)
            col_names.append(f"{image_id}_b{k+1}_y")

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
        return np.empty((0,), dtype=np.float64), []

    return residual_from_two_dfs(real_df, pred_df, cfg["K"])

# -----------------------------------------------------------------------
# Combined-geometry residual: concatenate residuals from every scenario.
# Each scenario contributes an independent set of projections; their
# residuals are stacked into a single vector so the optimizer sees all
# geometries simultaneously.
# Column names are prefixed with the scenario name to keep them unique.
# -----------------------------------------------------------------------

def build_combined_residual(theta, scenarios_data, work_dir, debug=True):
    """
    scenarios_data: list of dicts, each with keys
        name       – scenario identifier (str)
        real_df    – measured bead positions DataFrame
        angles_deg – projection angles (np.ndarray)
        cfg        – geometry/detector config dict
    Returns concatenated (r_vec, col_names) or (empty, []) on failure.
    """
    work_dir = Path(work_dir)
    r_parts, col_parts = [], []
    for sd in scenarios_data:
        pred_dir = work_dir / sd["name"]
        r, cols = build_residual_image_based(
            theta, sd["real_df"], sd["angles_deg"], sd["cfg"], pred_dir, debug=debug
        )
        if len(r) == 0:
            return np.empty((0,), dtype=np.float64), []
        r_parts.append(r)
        col_parts.extend(f"{sd['name']}_{c}" for c in cols)
    return np.concatenate(r_parts), col_parts


def numerical_jacobian_combined(theta, active_mask, scenarios_data, eps, work_dir):
    """
    Numerical Jacobian over the combined residual from all geometry scenarios.
    work_dir subdirectories are overwritten each iteration (disk stays bounded).
    """
    work_dir = Path(work_dir)
    r0, cols = build_combined_residual(theta, scenarios_data, work_dir / "base", debug=True)
    if len(r0) == 0:
        return None, None, None

    M = r0.size
    active_idx = np.where(active_mask)[0]
    P = active_idx.size
    J = np.zeros((M, P), dtype=np.float64)

    for col, j in enumerate(active_idx):
        t_p = theta.copy(); t_m = theta.copy()
        t_p[j] += eps[j];   t_m[j] -= eps[j]

        r_p, _ = build_combined_residual(t_p, scenarios_data, work_dir / f"p_{j:02d}", debug=False)
        r_m, _ = build_combined_residual(t_m, scenarios_data, work_dir / f"m_{j:02d}", debug=False)
        if len(r_p) == 0 or len(r_m) == 0:
            continue
        J[:, col] = (r_p - r_m) / (2.0 * eps[j])

    return r0, J, cols


def lm_solve_combined(scenarios_data, n_iters=200, lam=1e-2,
                      fix_source=False, fix_detector=False,
                      fix_object=False, fix_offset=False, fix_det_z=False,
                      work_dir="lm_work"):
    """
    LM optimisation over a joint residual built from all geometry scenarios.
    A single theta is shared and updated to explain observations across every
    scenario simultaneously.
    """
    work_dir = Path(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    theta = np.zeros(11, dtype=np.float64)
    eps   = np.full(11, 0.01, dtype=np.float64)

    active_mask    = make_active_mask(fix_source, fix_detector, fix_object, fix_alpha=False, fix_offset=fix_offset, fix_det_z=fix_det_z)
    dtheta_norm_hist = []
    stall_count    = 0
    df_r0          = None

    for it in range(200):
        print_theta_table(theta, it)

        # Print geometry for the first scenario only (representative)
        sd0 = scenarios_data[0]
        src_w, obj_w, det_w, _, _, _ = apply_theta_to_geometry(
            theta,
            src_world=sd0["cfg"]["SRC_WORLD"],
            obj_world=sd0["cfg"]["OBJ_WORLD"],
            det_world=sd0["cfg"]["DET_WORLD"],
        )
        print_unity_geometry(src_w, obj_w, det_w, sd0["angles_deg"][0])

        # Combined residual + Jacobian (overwrites same subdirs each iter)
        r, J, cols = numerical_jacobian_combined(theta, active_mask, scenarios_data, eps, work_dir / "jac")
        if r is None:
            print("Combined residual failed, aborting.")
            break
        r1 = np.asarray(r, dtype=np.float64).reshape(-1)
        print(f"Combined residual length: {r1.size} ({len(scenarios_data)} geometries), Jacobian: {J.shape}")

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

        r_new, cols_new = build_combined_residual(new_theta, scenarios_data, work_dir / "trial", debug=False)
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
        df_r0.to_csv(work_dir / "residual_history.csv", index=False)

        diff_sum = float(abs(theta - FAKE_THETA).sum())
        print(f"\niter {it:02d} cost={cost:.6f} -> {cost_new:.6f} |dtheta|={np.linalg.norm(dtheta):.6e}  lambda={lam:.3e}  diff_sum={diff_sum:.6f}")

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
    df_r0.to_csv(work_dir / "residual_history.csv", index=False)
    return theta, f"{np.linalg.norm(dtheta):.6e}", cost_new, it + 1


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gauss-Newton calibration — combined geometry (fixed fake_theta)")
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
        help="Subset of scenarios to include in the combined residual. Example: 'G0,G1'",
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
        help="Lambda variant to run. Example: 'GN', 'LM_low', 'LM_normal', 'LM_high'",
    )
    args = parser.parse_args()

    astra_scaling = 1

    DET_ROW = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    DET_COL = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Unpack offsets from FAKE_THETA to derive real-world positions.
    # real_OBJ_WORLD = unity_OBJ_WORLD + [dOx, dOy, dOz]
    # real_DET_WORLD = unity_DET_WORLD + [dDx, dDy, dDz]
    _dOx, _dOy, _dOz = FAKE_THETA[2], FAKE_THETA[3], FAKE_THETA[4]
    _dDx, _dDy, _dDz = FAKE_THETA[5], FAKE_THETA[6], FAKE_THETA[7]
    _alpha            = float(FAKE_THETA[8])

    GEOM_SCENARIOS = [
        # {
        #     # G0 — baseline, moderate spread | SDD=674
        #     "name": "G0",
        #     "SRC_WORLD":       np.array([0.0,   45.0,  0.0],   dtype=np.float32),
        #     "unity_OBJ_WORLD": np.array([0.0,   40.0,  589.0], dtype=np.float32),
        #     "unity_DET_WORLD": np.array([0.0,   45.0,  674.0], dtype=np.float32),
        #     # real_OBJ = [10-10,  15+5,  589-19] = [0,   20,  570]
        #     # real_DET = [0+2,    0-3,   674+0]  = [2,   -3,  674]  SDD=674
        # },
        {
            # G1 — source shifted right, detector near | SDD=720
            "name": "G1",
            "SRC_WORLD":       np.array([0.0,   55.0,  0.0],   dtype=np.float32),
            "unity_OBJ_WORLD": np.array([-10.0, 40.0,  589.0], dtype=np.float32),
            "unity_DET_WORLD": np.array([0.0,   55.0,  674.0], dtype=np.float32),
            # real_OBJ = [6-10,  28+5,  577-19] = [-4,  33,  558]
            # real_DET = [-5+2,  10-3,  720+0]  = [-3,   7,  720]  SDD=720
        },
        # {
        #     # G2 — source left, detector high | SDD=760
        #     "name": "G2",
        #     "SRC_WORLD":       np.array([0.0,   55.0,  0.0],   dtype=np.float32),
        #     "unity_OBJ_WORLD": np.array([10.0,  40.0,  589.0], dtype=np.float32),
        #     "unity_DET_WORLD": np.array([0.0,   55.0,  674.0], dtype=np.float32),
        #     # real_OBJ = [15-10, 42+5,  582-19] = [5,   47,  563]
        #     # real_DET = [8+2,   20-3,  760+0]  = [10,  17,  760]  SDD=760
        # },
        # {
        #     # G3 — source high, long throw | SDD=800
        #     "name": "G3",
        #     "SRC_WORLD":       np.array([0.0,   35.0,  0.0],   dtype=np.float32),
        #     "unity_OBJ_WORLD": np.array([-10.0, 40.0,  589.0], dtype=np.float32),
        #     "unity_DET_WORLD": np.array([0.0,   35.0,  674.0], dtype=np.float32),
        #     # real_OBJ = [18-10, 23+5,  580-19] = [8,   28,  561]
        #     # real_DET = [-8+2,  -5-3,  800+0]  = [-6,  -8,  800]  SDD=800
        # },
        # {
        #     # G4 — source low, compact geometry | SDD=690
        #     "name": "G4",
        #     "SRC_WORLD":       np.array([0.0,   35.0,  0.0],   dtype=np.float32),
        #     "unity_OBJ_WORLD": np.array([10.0,  40.0,  589.0], dtype=np.float32),
        #     "unity_DET_WORLD": np.array([0.0,   35.0,  674.0], dtype=np.float32),
        #     # real_OBJ = [7-10,  20+5,  576-19] = [-3,  25,  557]
        #     # real_DET = [10+2,  3-3,   690+0]  = [12,   0,  690]  SDD=690
        # },
        # {
        #     # G1 — source shifted right, detector near | SDD=720
        #     "name": "G5",
        #     "SRC_WORLD":       np.array([0.0,   55.0,  0.0],   dtype=np.float32),
        #     "unity_OBJ_WORLD": np.array([-10.0, 40.0,  589.0], dtype=np.float32),
        #     "unity_DET_WORLD": np.array([0.0,   55.0,  620.0], dtype=np.float32),
        #     # real_OBJ = [6-10,  28+5,  577-19] = [-4,  33,  558]
        #     # real_DET = [-5+2,  10-3,  720+0]  = [-3,   7,  720]  SDD=720
        # },
        # {
        #     # G2 — source left, detector high | SDD=760
        #     "name": "G6",
        #     "SRC_WORLD":       np.array([0.0,   55.0,  0.0],   dtype=np.float32),
        #     "unity_OBJ_WORLD": np.array([10.0,  40.0,  589.0], dtype=np.float32),
        #     "unity_DET_WORLD": np.array([0.0,   55.0,  620.0], dtype=np.float32),
        #     # real_OBJ = [15-10, 42+5,  582-19] = [5,   47,  563]
        #     # real_DET = [8+2,   20-3,  760+0]  = [10,  17,  760]  SDD=760
        # },
        # {
        #     # G3 — source high, long throw | SDD=800
        #     "name": "G7",
        #     "SRC_WORLD":       np.array([0.0,   35.0,  0.0],   dtype=np.float32),
        #     "unity_OBJ_WORLD": np.array([-10.0, 40.0,  589.0], dtype=np.float32),
        #     "unity_DET_WORLD": np.array([0.0,   35.0,  620.0], dtype=np.float32),
        #     # real_OBJ = [18-10, 23+5,  580-19] = [8,   28,  561]
        #     # real_DET = [-8+2,  -5-3,  800+0]  = [-6,  -8,  800]  SDD=800
        # },
        # {
        #     # G4 — source low, compact geometry | SDD=690
        #     "name": "G8",
        #     "SRC_WORLD":       np.array([0.0,   35.0,  0.0],   dtype=np.float32),
        #     "unity_OBJ_WORLD": np.array([10.0,  40.0,  589.0], dtype=np.float32),
        #     "unity_DET_WORLD": np.array([0.0,   35.0,  620.0], dtype=np.float32),
        #     # real_OBJ = [7-10,  20+5,  576-19] = [-3,  25,  557]
        #     # real_DET = [10+2,  3-3,   690+0]  = [12,   0,  690]  SDD=690
        # },
    ]

    # Derive real-world positions once from unity positions + FAKE_THETA offsets
    for sc in GEOM_SCENARIOS:
        sc["real_OBJ_WORLD"] = sc["unity_OBJ_WORLD"] + np.array([_dOx, _dOy, _dOz], dtype=np.float32)
        sc["real_DET_WORLD"] = sc["unity_DET_WORLD"] + np.array([_dDx, _dDy, _dDz], dtype=np.float32)

    BEAD_LIST    = list(range(1, 8))
    ANGLE_FACTORS = [3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360]

    LAMBDA_VALUES = [
        {"name": "GN",        "lam": 0.0},
        {"name": "LM_low",    "lam": 1e-4},
        {"name": "LM_normal", "lam": 1e-2},
        {"name": "LM_high",   "lam": 1.0},
    ]

    CUBOID_SIZES = [
        {"name": "compact",  "width": 10.0, "breadth": 15.0, "height": 20.0},
        {"name": "small",    "width": 20.0, "breadth": 22.0, "height": 40.0},
        {"name": "normal",   "width": 20.0, "breadth": 40.0, "height": 60.0},
        {"name": "square",   "width": 30.0, "breadth": 30.0, "height": 30.0},
        {"name": "tall",     "width": 10.0, "breadth": 15.0, "height": 80.0},
        {"name": "wide",     "width": 80.0, "breadth": 80.0, "height": 20.0},
        {"name": "coplanar", "width": 40.0, "breadth": 60.0, "height": 10.0},
    ]

    cli_angles = parse_int_list(args.angles)
    cli_beads  = parse_int_list(args.k)
    if cli_angles is not None:
        ANGLE_FACTORS = cli_angles
    if cli_beads is not None:
        BEAD_LIST = cli_beads

    # if args.scenario is not None:
    #     selected = set(args.scenario.replace(",", " ").split())
    #     GEOM_SCENARIOS = [sc for sc in GEOM_SCENARIOS if sc["name"] in selected]

    if args.cuboid is not None:
        CUBOID_SIZES = [cs for cs in CUBOID_SIZES if cs["name"] == args.cuboid]

    if args.lam is not None:
        LAMBDA_VALUES = [lv for lv in LAMBDA_VALUES if lv["name"] == args.lam]

    MIN_AREA   = 200
    MAX_AREA   = 6000
    VOXEL_SIZE = 0.1

    BASE_REAL_DIR = HERE / "simulated/real_scans_geom"

    initial_calibration = np.array([
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
    ])

    fake_theta_f32 = np.round(FAKE_THETA.astype(np.float32), 2)

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
                    scenario_names_str = "+".join(sc["name"] for sc in GEOM_SCENARIOS)
                    print("\n" + "#" * 80)
                    print(f"Running cuboid={cuboid_name} scenarios=[{scenario_names_str}] N_ANGLES={each_angle}, K={each_k}")
                    print(f"  FAKE_THETA (same for all): {fake_theta_f32}")

                    ANGLE_DEGREES = np.linspace(0.0, 360.0, each_angle, endpoint=False)

                    # ---- Generate real projections and build scenarios_data ----
                    scenarios_data = []
                    for sc in GEOM_SCENARIOS:
                        scenario_name   = sc["name"]
                        SRC_WORLD       = sc["SRC_WORLD"]
                        UNITY_OBJ_WORLD = sc["unity_OBJ_WORLD"]
                        UNITY_DET_WORLD = sc["unity_DET_WORLD"]
                        REAL_OBJ_WORLD  = sc["real_OBJ_WORLD"]
                        REAL_DET_WORLD  = sc["real_DET_WORLD"]

                        real_out_dir = BASE_REAL_DIR / cuboid_name / each_lambda["name"] / f"K{each_k}_N{each_angle}" / scenario_name
                        real_out_dir.mkdir(parents=True, exist_ok=True)

                        fetch_and_save_projections(
                            out_dir=real_out_dir,
                            src_world=SRC_WORLD,
                            obj_world=REAL_OBJ_WORLD,
                            det_world_base=REAL_DET_WORLD,
                            alpha=_alpha,
                            angles_deg=ANGLE_DEGREES,
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
                            phantom_name=HERE / f"phantoms/cuboid_phantom_{each_k}_{cuboid_name}.npy",
                        )

                        real_proj = build_wide_df_from_folder(
                            real_out_dir, K=each_k, min_area=MIN_AREA, max_area=MAX_AREA
                        )
                        print(f"  [{scenario_name}] real_proj rows: {len(real_proj)}")

                        cfg = {
                            "K": each_k,
                            "cuboid_name": cuboid_name,
                            "det_h": 760,
                            "det_w": 956,
                            "astra_scaling": astra_scaling,
                            "DET_SPACING": 0.149600,
                            "SRC_WORLD": SRC_WORLD,
                            "OBJ_WORLD": UNITY_OBJ_WORLD,   # optimizer starts from unity position
                            "DET_WORLD": UNITY_DET_WORLD,   # optimizer starts from unity position
                            "VOXEL_SIZE": VOXEL_SIZE,
                            "DET_COL": DET_COL,
                            "DET_ROW": DET_ROW,
                            "min_area": MIN_AREA,
                            "max_area": MAX_AREA,
                            "initial_calibration": initial_calibration,
                            "box_images": True,
                        }

                        scenarios_data.append({
                            "name":       scenario_name,
                            "real_df":    real_proj,
                            "angles_deg": ANGLE_DEGREES,
                            "cfg":        cfg,
                        })

                    # ---- Single joint optimisation over all scenarios ----
                    lambda_name = each_lambda["name"]
                    work_dir    = HERE / f"simulated/trial_geom_combined/{lambda_name}/{cuboid_name}/{each_k}_{each_angle}"

                    theta_hat, dtheta, cost, it = lm_solve_combined(
                        scenarios_data,
                        n_iters=50, lam=each_lambda["lam"],
                        fix_source=True, fix_detector=False,
                        fix_object=False, fix_offset=False,
                        fix_det_z=True,
                        work_dir=work_dir,
                    )

                    theta_minus_fake = theta_hat - fake_theta_f32
                    diff_sum         = float(abs(theta_minus_fake).sum())

                    os.makedirs(HERE / "simulated/theta_log_geom_combined", exist_ok=True)
                    THETA_TXT = HERE / f"simulated/theta_log_geom_combined/theta_hat_{lambda_name}_{cuboid_name}_{each_k}_{each_angle}.txt"
                    with open(THETA_TXT, "a") as ftxt:
                        ftxt.write(f"# lambda={lambda_name} cuboid={cuboid_name} scenarios=[{scenario_names_str}] N_ANGLES={each_angle}, K={each_k}\n")
                        ftxt.write("# fake_theta (fixed)\n")
                        ftxt.write(" ".join(f"{v:.2f}" for v in fake_theta_f32) + "\n")
                        ftxt.write("# theta_hat\n")
                        ftxt.write(" ".join(f"{v:.3f}" for v in theta_hat) + "\n")
                        ftxt.write("# Diff from Expected\n")
                        ftxt.write(" ".join(f"{v:.3f}" for v in theta_minus_fake) + "\n")
                        ftxt.write(f"# sum = {diff_sum:.3f}\n\n")

                    print("Fake theta (fixed):", fake_theta_f32)
                    print("Final estimated theta:", theta_hat)
                    print("Diff from Expected:", theta_minus_fake)
                    print("#" * 80 + "\n")
