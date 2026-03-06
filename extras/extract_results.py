from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


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

RE_DELTA_BLOCK = re.compile(
    r"delta offsets \(Unity frame\):\s*"
    r"Source\s*:\s*dSx=\s*([0-9eE+.\-]+),\s*dSy=\s*([0-9eE+.\-]+),\s*dSz=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Object\s*:\s*dOx=\s*([0-9eE+.\-]+),\s*dOy=\s*([0-9eE+.\-]+),\s*dOz=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Object offset:\s*offset_x=\s*([0-9eE+.\-]+),\s*offset_z=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Detector\s*:\s*dDx=\s*([0-9eE+.\-]+),\s*dDy=\s*([0-9eE+.\-]+),\s*dDz=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Obj Stage rotY\s*:\s*([0-9eE+.\-]+)\s*deg",
    re.MULTILINE
)

RE_ESTIMATED_DELTA_BLOCK = re.compile(
    r"Estimated delta:\s*"
    r"(.*?)"
    r"(?:Final cost table:|Final estimated delta:)",
    re.DOTALL
)

RE_ESTIMATED_DELTA_LINE = re.compile(
    r"^\s*(dSx|dSy|dSz|dOx|dOy|dOz|dDx|dDy|dDz|alpha|offset_x|offset_z):\s*([+\-]?[0-9eE+.\-]+)",
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


def _last_match(pattern: re.Pattern, text: str):
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


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
    row["final_iteration_header"] = iter_headers[-1] if iter_headers else None

    # Last iter summary line
    iter_lines = list(RE_ITER_LINE.finditer(text))
    if iter_lines:
        m = iter_lines[-1]
        row["final_iter_line_iter"] = int(m.group(1))
        row["cost_before"] = _to_float(m.group(2))
        row["cost_after"] = _to_float(m.group(3))
        row["ddelta"] = _to_float(m.group(4))
        row["lambda"] = _to_float(m.group(5))

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
    else:
        # fallback: last unity geometry block anywhere
        unity_blocks = list(RE_UNITY_GEOM_BLOCK.finditer(text))
        if unity_blocks:
            m = unity_blocks[-1]
            vals = list(map(_to_float, m.groups()))
            (
                row["src_x"], row["src_y"], row["src_z"],
                row["obj_x"], row["obj_y"], row["obj_z"],
                row["det_x"], row["det_y"], row["det_z"],
                row["obj_rotY_deg_after_calib"],
            ) = vals

    # Final "Estimated delta:" named block
    m = RE_ESTIMATED_DELTA_BLOCK.search(text)
    if m:
        delta_text = m.group(1)
        for mm in RE_ESTIMATED_DELTA_LINE.finditer(delta_text):
            key = mm.group(1)
            row[f"final_{key}"] = _to_float(mm.group(2))

    # Final array form, if present
    m = RE_FINAL_ESTIMATED_DELTA_ARRAY.search(text)
    if m:
        arr = [float(x) for x in m.group(1).replace("\n", " ").split()]
        row["final_estimated_delta_array"] = arr

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
        "final_iteration_header", "final_iter_line_iter",
        "cost_before", "cost_after", "ddelta", "lambda",
        "dSx", "dSy", "dSz", "dOx", "dOy", "dOz",
        "offset_x", "offset_z", "dDx", "dDy", "dDz", "stage_rotY_deg",
        "src_x", "src_y", "src_z",
        "obj_x", "obj_y", "obj_z",
        "det_x", "det_y", "det_z",
        "obj_rotY_deg_after_calib",
        "path", "parse_error",
    ]

    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    return df[cols]


if __name__ == "__main__":
    # change this
    folder = r"/vol/home/s3777103/Documents/workspace/Thesis/AutoCalibration/logs/hp_test_10"

    df = parse_log_folder(folder, pattern="*.log")

    print(df.to_string(index=False))

    out_csv = Path(folder) / "calibration_log_summary.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_csv}")