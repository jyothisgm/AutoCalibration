from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

LOG_DIR = Path(__file__).resolve().parent.parent / "logs_sim" / "hp_test_6"

# ----------------------------
# Regex helpers
# ----------------------------

RE_FILENAME = re.compile(r"run(\d+)_u(\d+)\.log$")

RE_SCENARIO = re.compile(
    r"Running cuboid=(\S+)\s+scenarios=\[([^\]]+)\]\s+N_ANGLES=(\d+),\s*K=(\d+)"
)

RE_ITER_HEADER = re.compile(r"^Iteration\s+(\d+)\s*$", re.MULTILINE)

RE_ITER_LINE = re.compile(
    r"^iter\s+(\d+)\s+cost=([0-9eE+.\-]+)\s*->\s*([0-9eE+.\-]+)"
    r"\s+\|dtheta\|=([0-9eE+.\-]+)\s+lambda=([0-9eE+.\-]+)"
    r"(?:\s+diff_sum=([0-9eE+.\-]+))?",
    re.MULTILINE,
)

RE_UNITY_GEOM_BLOCK = re.compile(
    r"Unity geometry \(world coordinates\):\s*"
    r"Source\s*:\s*x=\s*([0-9eE+.\-]+),\s*y=\s*([0-9eE+.\-]+),\s*z=\s*([0-9eE+.\-]+)\s*"
    r"Object\s*:\s*x=\s*([0-9eE+.\-]+),\s*y=\s*([0-9eE+.\-]+),\s*z=\s*([0-9eE+.\-]+)\s*"
    r"Detector\s*:\s*x=\s*([0-9eE+.\-]+),\s*y=\s*([0-9eE+.\-]+),\s*z=\s*([0-9eE+.\-]+)\s*"
    r"Obj rotY\s*:\s*([0-9eE+.\-]+)\s*deg",
    re.MULTILINE,
)

RE_THETA_BLOCK = re.compile(
    r"[Tt]heta offsets \(Unity frame\):\s*"
    r"Source\s*:\s*dSx=\s*([0-9eE+.\-]+),\s*dSy=\s*([0-9eE+.\-]+),\s*dSz=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Object\s*:\s*dOx=\s*([0-9eE+.\-]+),\s*dOy=\s*([0-9eE+.\-]+),\s*dOz=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Object offset:\s*offset_x=\s*([0-9eE+.\-]+),\s*offset_z=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Detector\s*:\s*dDx=\s*([0-9eE+.\-]+),\s*dDy=\s*([0-9eE+.\-]+),\s*dDz=\s*([0-9eE+.\-]+).*?\n"
    r"\s*Obj Stage rotY\s*:\s*([0-9eE+.\-]+)\s*deg",
    re.MULTILINE,
)

RE_FAKE_THETA = re.compile(r"Fake theta[^:]*:\s*\[([^\]]+)\]", re.DOTALL)
RE_FINAL_THETA = re.compile(r"Final estimated theta:\s*\[([^\]]+)\]", re.DOTALL)
RE_DIFF = re.compile(r"Diff from Expected:\s*\[([^\]]+)\]", re.DOTALL)


def _parse_array(s: str) -> list[float]:
    return [float(v) for v in s.replace("\n", " ").split()]


def parse_log_file(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")

    row: dict = {"file": path.name}

    # Filename metadata
    m_fn = RE_FILENAME.search(path.name)
    if m_fn:
        row["run_id"] = int(m_fn.group(1))
        row["phantom_id"] = m_fn.group(2)

    # Header metadata
    m = RE_SCENARIO.search(text)
    if m:
        row["cuboid"] = m.group(1)
        row["scenarios"] = m.group(2)          # e.g. "G1+G2+G3+G4+G5+G6+G7+G8"
        row["N_ANGLES"] = int(m.group(3))
        row["K"] = int(m.group(4))

    # Iteration count
    iter_headers = [int(h.group(1)) for h in RE_ITER_HEADER.finditer(text)]
    row["total_iters"] = iter_headers[-1] if iter_headers else None

    # Iter summary lines
    iter_lines = list(RE_ITER_LINE.finditer(text))
    if iter_lines:
        first_m, last_m = iter_lines[0], iter_lines[-1]
        ci = float(first_m.group(2))
        cf = float(last_m.group(2))
        di = float(first_m.group(4))
        df_ = float(last_m.group(4))
        row["cost_initial"] = ci
        row["cost_final"] = cf
        row["cost_change_pct"] = 100.0 * (ci - cf) / ci if ci != 0 else None
        row["ddtheta_initial"] = di
        row["dtheta"] = df_
        row["dtheta_change_pct"] = 100.0 * (di - df_) / di if di != 0 else None
        row["lambda"] = float(last_m.group(5))
        if last_m.group(6) is not None:
            row["diff_sum"] = float(last_m.group(6))

    # Unity geometry blocks
    unity_blocks = list(RE_UNITY_GEOM_BLOCK.finditer(text))
    if unity_blocks:
        vals = list(map(float, unity_blocks[0].groups()))
        (
            row["init_src_x"], row["init_src_y"], row["init_src_z"],
            row["init_obj_x"], row["init_obj_y"], row["init_obj_z"],
            row["init_det_x"], row["init_det_y"], row["init_det_z"],
            row["init_obj_rotY_deg"],
        ) = vals
        vals = list(map(float, unity_blocks[-1].groups()))
        (
            row["src_x"], row["src_y"], row["src_z"],
            row["obj_x"], row["obj_y"], row["obj_z"],
            row["det_x"], row["det_y"], row["det_z"],
            row["obj_rotY_deg_after_calib"],
        ) = vals

    # Last theta offsets block
    theta_blocks = list(RE_THETA_BLOCK.finditer(text))
    if theta_blocks:
        vals = list(map(float, theta_blocks[-1].groups()))
        (
            row["dSx"], row["dSy"], row["dSz"],
            row["dOx"], row["dOy"], row["dOz"],
            row["offset_x"], row["offset_z"],
            row["dDx"], row["dDy"], row["dDz"],
            row["stage_rotY_deg"],
        ) = vals

    # Fake theta
    m_fake = RE_FAKE_THETA.search(text)
    if m_fake:
        row["fake_theta"] = _parse_array(m_fake.group(1))

    # Final estimated theta array
    m_hat = RE_FINAL_THETA.search(text)
    if m_hat:
        arr = _parse_array(m_hat.group(1))
        row["final_estimated_theta_array"] = arr
        row["final_theta_abs_sum"] = sum(abs(v) for v in arr)

    # Diff from expected
    m_diff = RE_DIFF.search(text)
    if m_diff:
        arr = _parse_array(m_diff.group(1))
        row["diff_array"] = arr
        row["diff_abs_sum"] = sum(abs(v) for v in arr)

    return row


def parse_log_folder(folder: str | Path = LOG_DIR, pattern: str = "run*.log") -> pd.DataFrame:
    folder = Path(folder)
    rows = []
    for path in sorted(folder.rglob(pattern)):
        try:
            rows.append(parse_log_file(path))
        except Exception as e:
            rows.append({"file": path.name, "parse_error": str(e)})

    df = pd.DataFrame(rows)

    preferred_cols = [
        "file", "run_id", "phantom_id", "cuboid", "scenarios", "N_ANGLES", "K",
        "init_src_x", "init_src_y", "init_src_z",
        "init_obj_x", "init_obj_y", "init_obj_z",
        "init_det_x", "init_det_y", "init_det_z",
        "init_obj_rotY_deg",
        "total_iters",
        "cost_initial", "cost_final", "cost_change_pct",
        "ddtheta_initial", "dtheta", "dtheta_change_pct", "lambda",
        "diff_sum",
        "dSx", "dSy", "dSz",
        "dOx", "dOy", "dOz",
        "offset_x", "offset_z",
        "dDx", "dDy", "dDz",
        "stage_rotY_deg",
        "src_x", "src_y", "src_z",
        "obj_x", "obj_y", "obj_z",
        "det_x", "det_y", "det_z",
        "obj_rotY_deg_after_calib",
        "fake_theta", "final_estimated_theta_array", "final_theta_abs_sum",
        "diff_array", "diff_abs_sum",
        "parse_error",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [
        c for c in df.columns if c not in preferred_cols
    ]
    return df[cols]


if __name__ == "__main__":
    df = parse_log_folder()
    print(f"Parsed {len(df)} rows")

    display_cols = [
        "file", "run_id", "phantom_id", "cuboid", "scenarios", "N_ANGLES", "K",
        "total_iters", "cost_initial", "cost_final", "cost_change_pct",
        "ddtheta_initial", "dtheta", "lambda", "diff_sum",
        "dOx", "dOy", "dOz", "dDx", "dDy", "stage_rotY_deg",
        "final_theta_abs_sum", "diff_abs_sum",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))

    out_csv = LOG_DIR / "calibration_log_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
