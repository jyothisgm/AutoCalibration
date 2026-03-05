import numpy as np
import os
from datetime import datetime, timedelta

def write_scan_settings_txt(
    out_dir: str,
    image_width: int,
    image_height: int,
    voxel_size: float,
    det_spacing: float,
    src_world: np.ndarray,
    obj_world: np.ndarray,
    det_world: np.ndarray,
    initial_calibration: np.ndarray,
    astra_scaling: float,
    angles_deg: np.ndarray,
):
    """
    Writes a FleX-ray-like 'scan settings.txt' using values already available
    in fetch_and_save_projections.

    initial_calibration: shape (3,3) where rows are [src_offset, obj_offset, det_offset]
    in Unity world units (same units as src_world/obj_world/det_world).
    """
    start_dt = datetime.now()
    duration_sec = max(1, int(0.1 * len(angles_deg)))
    stop_dt = start_dt + timedelta(seconds=duration_sec)

    src_world = np.asarray(src_world, dtype=np.float64).reshape(3)
    obj_world = np.asarray(obj_world, dtype=np.float64).reshape(3)
    det_world = np.asarray(det_world, dtype=np.float64).reshape(3)

    # --- APPLY initial calibration offsets (in Unity/world units) ---
    ic = np.asarray(initial_calibration, dtype=np.float64)
    if ic.shape != (3, 3):
        raise ValueError(f"initial_calibration must have shape (3,3). Got {ic.shape}.")

    src_world = src_world + ic[0]
    obj_world = obj_world + ic[1]
    det_world = det_world + ic[2]
    # --- END APPLY ---

    # Geometry-derived values (mm if astra_scaling converts Unity units->mm)
    SOD = float(np.linalg.norm((src_world - obj_world) * astra_scaling))
    SDD = float(np.linalg.norm((src_world - det_world) * astra_scaling))
    magnification = float(SDD / SOD) if SOD > 0 else 0.0

    # Consistent with your logs
    HC = float(image_width/2.0)
    VC = float(image_height/2.0)
    COR = float(obj_world[1])
    voxel_size = det_spacing / (magnification) * 1000

    start_angle = float(np.min(angles_deg)) if len(angles_deg) else 0.0
    if len(angles_deg) >= 2:
        step = float(angles_deg[1] - angles_deg[0])
        last_angle = 360.0 if abs((float(angles_deg[-1]) + step) - 360.0) < 1e-3 else float(np.max(angles_deg))
    else:
        last_angle = float(np.max(angles_deg)) if len(angles_deg) else 0.0

    def fmt_date(dt: datetime) -> str:
        return dt.strftime("%d/%m/%Y")

    def fmt_time(dt: datetime) -> str:
        return dt.strftime("%H:%M:%S")

    def fmt_duration(sec: int) -> str:
        mins = sec // 60
        if mins <= 1:
            return f"{sec} seconds"
        return f"{mins} minutes"

    scan_id = start_dt.strftime("%y%m%d_%H%M%S")
    pixel_size = float(det_spacing)

    path = os.path.join(out_dir, "scan settings.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
f"""ScanID : {scan_id}
BatchID :
Project :
START
date: {fmt_date(start_dt)}; time: {fmt_time(start_dt)}

STOP
date: {fmt_date(stop_dt)}; time: {fmt_time(stop_dt)}

SCAN DURATION : {fmt_duration(duration_sec)}
COR : {COR:.6f}
VC : {VC:.6f}
HC : {HC:.6f}
SDD : {SDD:.6f}
SOD : {SOD:.6f}
Voxel size : {voxel_size:.6f}
Magnification : {magnification:.6f}

Operator :
Sample name :
Sample owner :
Application area :
Sample size : 0.000000 mm
Comment :
Scanner name :
Scanner type :
Acquila version :

X-ray tube : xraysource
Tube voltage : 90.000000
Tube power : 49.500000
Vacuum level :
Filter :
Focus mode : microfocus
Target current : 0.000000
Filament status :
Filament mode :
Output mode : Emission current

Camera : detector
Exposure time (ms) : 99.998199
Number of averages : 10.000000
Original pixel size : {pixel_size/2:.6f}
Imaging mode :
Binning value : 2
Binned pixel size : {pixel_size:.6f}
ROI (LTRB) :

Script summary:

scan type:
smooth scan
# projections: {len(angles_deg)}
Start angle : {start_angle:.6f}
Last angle : {last_angle:.6f}
# pre flat fields: 0
# post flat fields: 0
# offset images: 0
Reference images every 0 projections
Preheating time: 0 minutes
Axis for flat field movement :
"""
        )