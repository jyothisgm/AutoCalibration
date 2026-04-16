from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

def get_scenario_number(s):
    if isinstance(s, str):
        m = re.search(r"\d+", s)
        if m:
            return int(m.group())
    return 9999

# ----------------------------
# Regex helpers
# ----------------------------

RE_ITER_HEADER = re.compile(r"^Iteration\s+(\d+)\s*$", re.MULTILINE)

RE_ITER_LINE = re.compile(
    r"iter\s+(\d+)\s+cost=([0-9eE+.\-]+)\s*->\s*([0-9eE+.\-]+)\s+\|dtheta\|=([0-9eE+.\-]+)\s+lambda=([0-9eE+.\-]+)"
)

RE_SCENARIO = re.compile(
    r"Running scenario=(.*?)\s+Projections=(\d+)\s+Used=(\d+),\s*K=(\d+)"
)

RE_SCENARIO_NEW = re.compile(
    r"Running cuboid=\S+\s+scenario=(\S+)\s+N_ANGLES=(\d+),\s*K=(\d+)"
)

RE_UNITY_GEOM_BLOCK = re.compile(
    r"Unity geometry \(world coordinates\):\s*"
    r"Source\s*:\s*x=\s*([0-9eE+.\-]+),\s*y=\s*([0-9eE+.\-]+),\s*z=\s*([0-9eE+.\-]+)\s*"
    r"Object\s*:\s*x=\s*([0-9eE+.\-]+),\s*y=\s*([0-9eE+.\-]+),\s*z=\s*([0-9eE+.\-]+)\s*"
    r"Detector\s*:\s*x=\s*([0-9eE+.\-]+),\s*y=\s*([0-9eE+.\-]+),\s*z=\s*([0-9eE+.\-]+)\s*"
    r"Obj rotY\s*:\s*([0-9eE+.\-]+)\s*deg",
    re.MULTILINE
)

RE_AFTER_CALIB_UNITY = re.compile(
    r"After Calibration:\s*"
    r"Unity geometry \(world coordinates\):\s*"
    r"Source\s*:\s*x=\s*([0-9eE+.\-]+),\s*y=\s*([0-9eE+.\-]+),\s*z=\s*([0-9eE+.\-]+)\s*"
    r"Object\s*:\s*x=\s*([0-9eE+.\-]+),\s*y=\s*([0-9eE+.\-]+),\s*z=\s*([0-9eE+.\-]+)\s*"
    r"Detector\s*:\s*x=\s*([0-9eE+.\-]+),\s*y=\s*([0-9eE+.\-]+),\s*z=\s*([0-9eE+.\-]+)\s*"
    r"Obj rotY\s*:\s*([0-9eE+.\-]+)\s*deg",
    re.MULTILINE
)

RE_INITIAL_CALIB_BLOCK = re.compile(
    r"Initial calibration:\s*"
    r"Source\s*:\s*x=\s*([0-9eE+.\-]+),\s*y=\s*([0-9eE+.\-]+),\s*z=\s*([0-9eE+.\-]+)\s*"
    r"Object\s*:\s*x=\s*([0-9eE+.\-]+),\s*y=\s*([0-9eE+.\-]+),\s*z=\s*([0-9eE+.\-]+)\s*"
    r"Detector\s*:\s*x=\s*([0-9eE+.\-]+),\s*y=\s*([0-9eE+.\-]+),\s*z=\s*([0-9eE+.\-]+)",
    re.MULTILINE
)

RE_THETA_BLOCK = re.compile(
    r"[Tt]heta offsets \(Unity frame\):\s*"
    r"Source\s*:\s*dSx=\s*([0-9eE+.\-]+),\s*dSy=\s*([0-9eE+.\-]+),\s*dSz=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Object\s*:\s*dOx=\s*([0-9eE+.\-]+),\s*dOy=\s*([0-9eE+.\-]+),\s*dOz=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Object offset:\s*offset_x=\s*([0-9eE+.\-]+),\s*offset_z=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Detector\s*:\s*dDx=\s*([0-9eE+.\-]+),\s*dDy=\s*([0-9eE+.\-]+),\s*dDz=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Obj Stage rotY\s*:\s*([0-9eE+.\-]+)\s*deg",
    re.MULTILINE
)

RE_FINAL_ESTIMATED_THETA_ARRAY = re.compile(
    r"Final estimated theta:\s*\[([^\]]+)\]",
    re.DOTALL
)


# ----------------------------
# Parsing helpers
# ----------------------------

def _to_float(x: str) -> float:
    return float(x.strip())


def parse_log_file(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")

    row: dict = {
        "file": path.name,
    }

    # Scenario metadata
    m = RE_SCENARIO.search(text)
    if m:
        row["scenario"] = m.group(1).strip()
        row["projections"] = int(m.group(2))
        row["used"] = int(m.group(3))
        row["K"] = int(m.group(4))
    else:
        m = RE_SCENARIO_NEW.search(text)
        if m:
            row["scenario"] = m.group(1).strip()
            row["used"] = int(m.group(2))
            row["projections"] = row["used"]
            row["K"] = int(m.group(3))

    # Last iteration header seen
    iter_headers = [int(m.group(1)) for m in RE_ITER_HEADER.finditer(text)]
    row["total_iters"] = iter_headers[-1] if iter_headers else None

    # Iter summary lines
    iter_lines = list(RE_ITER_LINE.finditer(text))
    if iter_lines:
        first_m = iter_lines[0]
        last_m = iter_lines[-1]

        # first-ever values
        row["cost_initial"] = _to_float(first_m.group(2))
        row["ddtheta_initial"] = _to_float(first_m.group(4))

        # final values
        row["cost_final"] = _to_float(last_m.group(2))
        row["dtheta"] = _to_float(last_m.group(4))
        row["lambda"] = _to_float(last_m.group(5))

    # Initial calibration block
    m = RE_INITIAL_CALIB_BLOCK.search(text)
    if m:
        vals = list(map(_to_float, m.groups()))
        (
            # row["src_x_cal"], row["src_y_cal"], row["src_z_cal"],
            _, _, _,
            row["obj_x_cal"], row["obj_y_cal"], row["obj_z_cal"],
            row["det_x_cal"], row["det_y_cal"], row["det_z_cal"],
        ) = vals
    # Initial Unity geometry block = first unity geometry block anywhere
    unity_blocks = list(RE_UNITY_GEOM_BLOCK.finditer(text))
    if unity_blocks:
        m0 = unity_blocks[0]
        vals0 = list(map(_to_float, m0.groups()))
        (
            row["init_src_x"], row["init_src_y"], row["init_src_z"],
            row["init_obj_x"], row["init_obj_y"], row["init_obj_z"],
            row["init_det_x"], row["init_det_y"], row["init_det_z"],
            row["init_obj_rotY_deg"],
        ) = vals0

    # Last theta offsets block
    theta_blocks = list(RE_THETA_BLOCK.finditer(text))
    if theta_blocks:
        m = theta_blocks[-1]
        vals = list(map(_to_float, m.groups()))
        (
            row["dSx"], row["dSy"], row["dSz"],
            row["dOx"], row["dOy"], row["dOz"],
            row["offset_x"], row["offset_z"],
            row["dDx"], row["dDy"], row["dDz"],
            row["stage_rotY_deg"],
        ) = vals

    # Last "After Calibration" unity geometry block
    after_blocks = list(RE_AFTER_CALIB_UNITY.finditer(text))
    if after_blocks:
        m = after_blocks[-1]
        vals = list(map(_to_float, m.groups()))
        (
            row["src_x"], row["src_y"], row["src_z"],
            row["obj_x"], row["obj_y"], row["obj_z"],
            row["det_x"], row["det_y"], row["det_z"],
            row["obj_rotY_deg_after_calib"],
        ) = vals
    elif unity_blocks:
        # fallback: last unity geometry block anywhere
        m = unity_blocks[-1]
        vals = list(map(_to_float, m.groups()))
        (
            row["src_x"], row["src_y"], row["src_z"],
            row["obj_x"], row["obj_y"], row["obj_z"],
            row["det_x"], row["det_y"], row["det_z"],
            row["obj_rotY_deg_after_calib"],
        ) = vals

    # Final array form, if present
    m = RE_FINAL_ESTIMATED_THETA_ARRAY.search(text)
    if m:
        arr = [float(x) for x in m.group(1).replace("\n", " ").split()]
        row["final_estimated_theta_array"] = arr
        row["final_theta_abs_sum"] = sum(abs(v) for v in arr)
    cost_initial = row["cost_initial"]
    cost_final = row["cost_final"]
    ddtheta_initial = row["ddtheta_initial"]
    dtheta = row["dtheta"]

    row["cost_change_pct"] = (
        100.0 * (cost_initial - cost_final) / cost_initial
        if cost_initial != 0 else None
    )

    row["dtheta_change_pct"] = (
        100.0 * (ddtheta_initial - dtheta) / ddtheta_initial
        if ddtheta_initial != 0 else None
    )
    return row


def parse_log_folder(folder: str | Path, pattern: str = "*.log") -> pd.DataFrame:
    folder = Path(folder)
    rows = []

    for path in sorted(folder.rglob(pattern)):
        try:
            rows.append(parse_log_file(path))
        except Exception as e:
            rows.append({
                "file": path.name,
                "path": str(path),
                "parse_error": str(e),
            })

    df = pd.DataFrame(rows)

    preferred_cols = [
        "file", "scenario", "projections", "used", "K",

        # "src_x_cal", "src_y_cal", "src_z_cal",
        "obj_x_cal", "obj_y_cal", "obj_z_cal",
        "det_x_cal", "det_y_cal", "det_z_cal",

        "init_src_x", "init_src_y", "init_src_z",
        "init_obj_x", "init_obj_y", "init_obj_z",
        "init_det_x", "init_det_y", "init_det_z",
        "init_obj_rotY_deg",

        "total_iters",
        "cost_initial", "cost_final", "cost_change_pct",
        "ddtheta_initial", "dtheta", "dtheta_change_pct", "lambda",

        "dSx", "dSy", "dSz",
        "dOx", "dOy", "dOz",
        "offset_x", "offset_z",
        "dDx", "dDy", "dDz",
        "stage_rotY_deg",

        "src_x", "src_y", "src_z",
        "obj_x", "obj_y", "obj_z",
        "det_x", "det_y", "det_z",
        "obj_rotY_deg_after_calib",

        "path", "parse_error",
    ]

    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    return df[cols]


# ----------------------------
# Heatmap
# ----------------------------

BRACKETS       = [0, 0.2, 0.5, 1, 3, 10, float("inf")]
BRACKET_LABELS = ["0–0.2", "0.2–0.5", "0.5–1", "1–3", "3–10", "10+"]
BRACKET_COLORS = ["#1a9641", "#74c476", "#a6d96a", "#ffffbf", "#fdae61", "#f46d43", "#d7191c"]


def _scan_group(scenario: str) -> str:
    """Return the baseline group for a scenario name."""
    n = get_scenario_number(scenario)
    if n == 1:
        return "scan1"
    elif 2 <= n <= 6:
        return "scan2_6"
    else:
        return "scan7_11"


def plot_scan_angle_heatmap(
    df: pd.DataFrame,
    out_path: Path,
    metric: str = "final_theta_abs_sum",
    baseline_used: int = 360,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    if df.empty:
        print("Heatmap skipped: empty dataframe")
        return
    if metric not in df.columns:
        print(f"Heatmap skipped: metric '{metric}' not found in dataframe")
        return
    if "scenario" not in df.columns or "used" not in df.columns:
        print("Heatmap skipped: requires 'scenario' and 'used' columns")
        return

    def to_bracket(val):
        if pd.isna(val):
            return float("nan")
        for i in range(len(BRACKETS) - 1):
            if BRACKETS[i] <= abs(val) < BRACKETS[i + 1]:
                return i
        return len(BRACKETS) - 2

    cmap = mcolors.ListedColormap(BRACKET_COLORS)
    norm = mcolors.BoundaryNorm(
        boundaries=[-0.5 + i for i in range(len(BRACKET_COLORS) + 1)],
        ncolors=len(BRACKET_COLORS),
    )

    # Compute grouped baselines at baseline_used projections
    df360 = df[df["used"] == baseline_used].copy()
    df360["_group"] = df360["scenario"].apply(_scan_group)
    group_baseline = df360.groupby("_group")[metric].mean()

    # Per-scenario baseline: Scan1 = its own 360 value; Scan2-6 = group mean; Scan7-11 = group mean
    scen_means_360 = df360.groupby("scenario")[metric].mean()

    def get_baseline(scenario):
        g = _scan_group(scenario)
        if g == "scan1":
            return scen_means_360.get(scenario, float("nan"))
        return group_baseline.get(g, float("nan"))

    # Mean metric per (scenario, used)
    grouped = df.groupby(["scenario", "used"])[metric].mean().reset_index()
    grouped["baseline"] = grouped["scenario"].apply(get_baseline)
    grouped["delta"] = grouped[metric] - grouped["baseline"]

    pivot = grouped.pivot(index="scenario", columns="used", values="delta")

    scenarios = sorted(pivot.index.tolist(), key=get_scenario_number)
    angles = sorted(pivot.columns.tolist())
    pivot = pivot.reindex(index=scenarios, columns=angles)

    bracket_grid = pivot.map(to_bracket).values.astype(float)

    fig, ax = plt.subplots(
        figsize=(max(9, len(angles) * 0.9 + 2), max(3, len(scenarios) * 0.6 + 2))
    )
    ax.imshow(bracket_grid, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(angles)))
    ax.set_xticklabels(angles, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=8)
    ax.set_xlabel("N (number of projections used)", fontsize=10)
    ax.set_ylabel("Scan", fontsize=10)
    ax.set_title(
        f"MAE delta vs {baseline_used}-projection baseline\n"
        r"(Scan 1: own baseline; Scans 2–6 \& 7–11: group mean baseline)",
        fontsize=11,
    )

    for ri, scen in enumerate(scenarios):
        for ci, n in enumerate(angles):
            val = pivot.loc[scen, n] if (scen in pivot.index and n in pivot.columns) else float("nan")
            if not pd.isna(val):
                b_idx = int(to_bracket(val))
                txt_color = "white" if b_idx in (0, 6) else "black"
                ax.text(ci, ri, f"{val:.3f}", ha="center", va="center", fontsize=6.5, color=txt_color)

    patches = [mpatches.Patch(color=BRACKET_COLORS[i], label=BRACKET_LABELS[i])
               for i in range(len(BRACKET_LABELS))]
    ax.legend(handles=patches, title="|delta|", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=8, title_fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap: {out_path}")


if __name__ == "__main__":
    folder = Path("/vol/home/s3777103/Documents/workspace/Thesis/AutoCalibration/logs/hp_test_4")
    folder = Path("/vol/home/s3777103/Documents/workspace/Thesis/AutoCalibration/results/real")
    # for folder in sorted(p for p in parent.iterdir() if p.is_dir()):
    df = parse_log_folder(folder, pattern="*.log")

    # ----------------------------
    # mean / std rows
    # ----------------------------

    d_cols = [
        "dSx", "dSy", "dSz",
        "dOx", "dOy", "dOz",
        "offset_x", "offset_z",
        "dDx", "dDy", "dDz",
        "stage_rotY_deg",
    ]
    if "scenario" in df.columns:
        df["scenario_num"] = df["scenario"].apply(get_scenario_number)
        df = df.sort_values(by=["scenario_num", "used"]).drop(columns="scenario_num").reset_index(drop=True)
        
    

    # keep only columns that exist
    # d_cols = [c for c in d_cols if c in df.columns]

    # mean_row = {}
    # std_row = {}

    # for c in d_cols:
    #     if c in ["offset_x", "offset_z", "stage_rotY_deg"]:
    #         scen_num = (
    #             df["scenario"]
    #             .astype(str)
    #             .str.extract(r"(\d+)")[0]
    #             .astype(float)
    #         )

    #         mask1 = scen_num.between(1, 6)
    #         mask2 = scen_num.between(7, 11)

    #         vals1 = pd.to_numeric(df.loc[mask1, c], errors="coerce")
    #         vals2 = pd.to_numeric(df.loc[mask2, c], errors="coerce")

    #         mean_row[c] = f"{vals1.mean():.6f} | {vals2.mean():.6f}"
    #         std_row[c]  = f"{vals1.std():.6f} | {vals2.std():.6f}"

    #     else:

    #         vals = pd.to_numeric(df[c], errors="coerce")

    #         mean_row[c] = vals.mean()
    #         std_row[c]  = vals.std()


    #     mean_row["file"] = "MEAN"
    #     std_row["file"] = "STD"

    #     df = pd.concat(
    #         [df, pd.DataFrame([mean_row]), pd.DataFrame([std_row])],
    #         ignore_index=True
    #     )

    # ----------------------------
    # Delta summary rows (mean & SD per scan group)
    # ----------------------------
    metric = "final_theta_abs_sum"
    if metric in df.columns and "scenario" in df.columns and "used" in df.columns:
        df360 = df[df["used"] == 360].copy()
        df360["_group"] = df360["scenario"].apply(_scan_group)
        group_baseline = df360.groupby("_group")[metric].mean()
        scen_means_360 = df360.groupby("scenario")[metric].mean()

        def get_baseline(scenario):
            g = _scan_group(scenario)
            if g == "scan1":
                return scen_means_360.get(scenario, float("nan"))
            return group_baseline.get(g, float("nan"))

        df["_delta"] = df[metric] - df["scenario"].apply(get_baseline)
        df["_group"] = df["scenario"].apply(_scan_group)

        group_labels = {
            "scan1":    "MEAN/SD Scan 1",
            "scan2_6":  "MEAN/SD Scan 2-6",
            "scan7_11": "MEAN/SD Scan 7-11",
        }
        d_cols = [c for c in [
            "dSx", "dSy", "dSz", "dOx", "dOy", "dOz",
            "offset_x", "offset_z", "dDx", "dDy", "dDz", "stage_rotY_deg",
        ] if c in df.columns]

        # restrict theta cols to used=360 rows only
        df360_theta = df360.copy()
        df360_theta["_group"] = df360_theta["scenario"].apply(_scan_group)

        summary_rows = []
        for gkey, glabel in group_labels.items():
            mask_all  = df["_group"] == gkey
            mask_360  = df360_theta["_group"] == gkey

            mean_row = {"file": f"MEAN {glabel}"}
            std_row  = {"file": f"SD   {glabel}"}

            # delta of final_theta_abs_sum (all N)
            delta_vals = df.loc[mask_all, "_delta"].dropna()
            mean_row[metric] = delta_vals.mean()
            std_row[metric]  = delta_vals.std()

            # theta component cols at used=360
            for c in d_cols:
                vals = pd.to_numeric(df360_theta.loc[mask_360, c], errors="coerce").dropna()
                mean_row[c] = vals.mean()
                std_row[c]  = vals.std()

            summary_rows.append(mean_row)
            summary_rows.append(std_row)

        df = df.drop(columns=["_delta", "_group"])
        df = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)

    out_csv = Path(folder) / "calibration_log_summary.csv"
    df.to_csv(out_csv, index=False)
    print("saved:", out_csv)

    out_heatmap = Path(folder) / "calibration_heatmap.png"
    plot_scan_angle_heatmap(df, out_heatmap, metric="final_theta_abs_sum", baseline_used=360)
