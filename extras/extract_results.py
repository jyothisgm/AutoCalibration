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
    r"iter\s+(\d+)\s+cost=([0-9eE+.\-]+)\s*->\s*([0-9eE+.\-]+)\s+\|ddelta\|=([0-9eE+.\-]+)\s+lambda=([0-9eE+.\-]+)"
)

RE_SCENARIO = re.compile(
    r"Running scenario=(.*?)\s+Projections=(\d+)\s+Used=(\d+),\s*K=(\d+)"
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

RE_DELTA_BLOCK = re.compile(
    r"delta offsets \(Unity frame\):\s*"
    r"Source\s*:\s*dSx=\s*([0-9eE+.\-]+),\s*dSy=\s*([0-9eE+.\-]+),\s*dSz=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Object\s*:\s*dOx=\s*([0-9eE+.\-]+),\s*dOy=\s*([0-9eE+.\-]+),\s*dOz=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Object offset:\s*offset_x=\s*([0-9eE+.\-]+),\s*offset_z=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Detector\s*:\s*dDx=\s*([0-9eE+.\-]+),\s*dDy=\s*([0-9eE+.\-]+),\s*dDz=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Obj Stage rotY\s*:\s*([0-9eE+.\-]+)\s*deg",
    re.MULTILINE
)

RE_FINAL_ESTIMATED_DELTA_ARRAY = re.compile(
    r"Final estimated delta:\s*\[([^\]]+)\]",
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
        row["dddelta_initial"] = _to_float(first_m.group(4))

        # final values
        row["cost_final"] = _to_float(last_m.group(2))
        row["ddelta"] = _to_float(last_m.group(4))
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

    # Last delta offsets block
    delta_blocks = list(RE_DELTA_BLOCK.finditer(text))
    if delta_blocks:
        m = delta_blocks[-1]
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
    m = RE_FINAL_ESTIMATED_DELTA_ARRAY.search(text)
    if m:
        arr = [float(x) for x in m.group(1).replace("\n", " ").split()]
        row["final_estimated_delta_array"] = arr
    cost_initial = row["cost_initial"]
    cost_final = row["cost_final"]
    dddelta_initial = row["dddelta_initial"]
    ddelta = row["ddelta"]

    row["cost_change_pct"] = (
        100.0 * (cost_initial - cost_final) / cost_initial
        if cost_initial != 0 else None
    )

    row["ddelta_change_pct"] = (
        100.0 * (dddelta_initial - ddelta) / dddelta_initial
        if dddelta_initial != 0 else None
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
        "dddelta_initial", "ddelta", "ddelta_change_pct", "lambda",

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


if __name__ == "__main__":
    for i in range(20, 23):
        folder = f"/vol/home/s3777103/Documents/workspace/Thesis/AutoCalibration/logs/hp_test_{i}"

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
        d_cols = [c for c in d_cols if c in df.columns]

        mean_row = {}
        std_row = {}

        for c in d_cols:
            if c in ["offset_x", "offset_z"]:
                scen_num = (
                    df["scenario"]
                    .astype(str)
                    .str.extract(r"(\d+)")[0]
                    .astype(float)
                )

                mask1 = scen_num.between(1, 6)
                mask2 = scen_num.between(7, 11)

                vals1 = pd.to_numeric(df.loc[mask1, c], errors="coerce")
                vals2 = pd.to_numeric(df.loc[mask2, c], errors="coerce")

                mean_row[c] = f"{vals1.mean():.6f} | {vals2.mean():.6f}"
                std_row[c]  = f"{vals1.std():.6f} | {vals2.std():.6f}"

            else:

                vals = pd.to_numeric(df[c], errors="coerce")

                mean_row[c] = vals.mean()
                std_row[c]  = vals.std()


        mean_row["file"] = "MEAN"
        std_row["file"] = "STD"

        df = pd.concat(
            [df, pd.DataFrame([mean_row]), pd.DataFrame([std_row])],
            ignore_index=True
        )

        out_csv = Path(folder) / "calibration_log_summary.csv"
        df.to_csv(out_csv, index=False)

        print("saved:", out_csv)