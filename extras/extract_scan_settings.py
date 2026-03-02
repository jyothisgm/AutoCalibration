
import os
import re
import numpy as np
from pathlib import Path

import pprint
from image_flip import to_astra_line_integrals

np.set_printoptions(suppress=True)

def extract_first_float(line):
    return float(re.search(r":\s*([-+]?\d*\.?\d+)", line).group(1))

def extract_roi(line):
    # ROI (LTRB) : 32,8,1943,1527
    values = re.search(r":\s*([0-9,\s]+)", line).group(1)
    return np.array([int(v.strip()) for v in values.split(",")], dtype=np.int32)

def parse_scan_settings(file_path):
    params = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            # Geometry
            if line.startswith("mag_obj"):
                params["mag_obj"] = extract_first_float(line)
            elif line.startswith("mag_det"):
                params["mag_det"] = extract_first_float(line)
            elif line.startswith("rot_obj"):
                params["rot_obj"] = extract_first_float(line)
            elif line.startswith("ver_obj"):
                params["ver_obj"] = extract_first_float(line)
            elif line.startswith("ver_tube"):
                params["ver_tube"] = extract_first_float(line)
            elif line.startswith("ver_det"):
                params["ver_det"] = extract_first_float(line)
            elif line.startswith("tra_obj"):
                params["tra_obj"] = extract_first_float(line)
            elif line.startswith("tra_tube"):
                params["tra_tube"] = extract_first_float(line)
            elif line.startswith("tra_det"):
                params["tra_det"] = extract_first_float(line)

            # Detector info
            elif line.startswith("Original pixel size"):
                params["original_pixel_size"] = extract_first_float(line)
            elif line.startswith("Binning value"):
                params["binning_value"] = int(extract_first_float(line))
            elif line.startswith("Binned pixel size"):
                params["binned_pixel_size"] = extract_first_float(line)
            elif line.startswith("ROI (LTRB)"):
                params["ROI"] = extract_roi(line)

    return params


def build_geometry_list(scan_root="scan"):
    GEOMETRY = []
    for root, dirs, files in os.walk(scan_root):
        if "scan settings.txt" in files:
            print(f"Checking folder: {root}")
            to_astra_line_integrals(root, f"{root}\\out_line_integrals")
            folder_name = os.path.basename(root)
            settings_path = os.path.join(root, "scan settings.txt")

            p = parse_scan_settings(settings_path)

            pprint.pp(p)
            pprint.pp(f"Detector Width (mm): {(p["ROI"][2] - p["ROI"][0]) / p["binning_value"]}")
            pprint.pp(f"Detector Height (mm): {(p["ROI"][3] - p["ROI"][1]) / p["binning_value"]}")

            geom = {
                "name": folder_name,
                "src": np.array(
                    [p["tra_tube"], p["ver_tube"], 0],
                    dtype=np.float32,
                ),
                "det": np.array(
                    [p["tra_det"], p["ver_det"], p["mag_det"]],
                    dtype=np.float32,
                ),
                "obj": np.array(
                    [p["tra_obj"], p["ver_obj"], p["mag_obj"]],
                    dtype=np.float32,
                ),
                "initial_angle_deg": p["rot_obj"],
            }

            GEOMETRY.append(geom)

    return GEOMETRY

if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    REAL_DIR = HERE / f"2026-02-19_Beads_phantom"
    GEOMETRY = build_geometry_list(REAL_DIR)
    pprint.pp(GEOMETRY)