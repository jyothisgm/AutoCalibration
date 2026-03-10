import cv2
import os
import shutil
import socket
import numpy as np

from PIL import Image
from astra_server import AstraServer
from extras.scan_settings import write_scan_settings_txt


def reset_folder(folder):
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)

def apply_napari_contrast_and_gamma(
    img,
    low_percentile: float = 99.0,   # lower contrast limit
    high_percentile: float = 100.0, # upper contrast limit (usually max)
    gamma: float = 0.2
):
    if img is None:
        raise ValueError("Could not read image")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)

    # ---- Compute contrast limits like napari ----
    low = np.percentile(img, low_percentile)
    high = np.percentile(img, high_percentile)

    if high <= low:
        raise ValueError("Invalid contrast limits")

    # ---- Clip + rescale (napari behavior) ----
    clipped = np.clip(img, low, high)
    norm = (clipped - low) / (high - low)

    # ---- Gamma correction ----
    gamma_corrected = np.power(norm, gamma)

    # Convert to 8-bit for saving
    out = (gamma_corrected * 255).astype(np.uint8)

    return out

def geom12_metrics(g):
    g = np.asarray(g, dtype=float).reshape(12)

    # unpack from ASTRA packed (x,z,y) back to true (x,y,z)
    S = unpack_xzy(g[0:3])
    D = unpack_xzy(g[3:6])
    u = unpack_xzy(g[6:9])
    v = unpack_xzy(g[9:12])

    sod = np.linalg.norm(S)
    sdd = np.linalg.norm(D - S)
    mag = sdd / sod

    r = D - S  # source -> detector
    r_hat = r / np.linalg.norm(r)

    n = np.cross(u, v)
    n_hat = n / np.linalg.norm(n)

    # Physical incidence: should be 0° whether normal is + or - (plane has two normals)
    cosang = float(np.dot(r_hat, n_hat))
    cosang = np.clip(cosang, -1.0, 1.0)
    inc_deg = float(np.degrees(np.arccos(abs(cosang))))

    return sod, sdd, mag, inc_deg

def rotate_y_xz(v: np.ndarray, cosY: float, sinY: float) -> np.ndarray:
    """
    Unity rotation around Y axis applied to x,z (y unchanged):
        x' = x*cos - z*sin
        z' = x*sin + z*cos
    """
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    x2 = x * cosY - z * sinY
    z2 = x * sinY + z * cosY
    return np.array([x2, y, z2], dtype=np.float32)

def pack_xzy(v: np.ndarray) -> np.ndarray:
    return np.array([v[0], v[2], v[1]], dtype=np.float32)

def unpack_xzy(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return np.array([v[0], v[2], v[1]], dtype=np.float64)

def print_unity_geometry(src_w, obj_w, det_w, rot_y_deg):
    print("\nUnity geometry (world coordinates):")
    print(f"  Source   : x={src_w[0]:8.3f}, y={src_w[1]:8.3f}, z={src_w[2]:8.3f}")
    print(f"  Object   : x={obj_w[0]:8.3f}, y={obj_w[1]:8.3f}, z={obj_w[2]:8.3f}")
    print(f"  Detector : x={det_w[0]:8.3f}, y={det_w[1]:8.3f}, z={det_w[2]:8.3f}")
    print(f"  Obj rotY : {rot_y_deg:8.4f} deg")

def print_initial_calibration(calib):
    print("\nInitial calibration:")
    print(f"  Source   : x={calib[0][0]:8.3f}, y={calib[0][1]:8.3f}, z={calib[0][2]:8.3f}")
    print(f"  Object   : x={calib[1][0]:8.3f}, y={calib[1][1]:8.3f}, z={calib[1][2]:8.3f}")
    print(f"  Detector : x={calib[2][0]:8.3f}, y={calib[2][1]:8.3f}, z={calib[2][2]:8.3f}")

def print_geometry_vector(geom12):
    g = np.asarray(geom12, dtype=float).reshape(12)
    src = g[0:3]
    det = g[3:6]
    u   = g[6:9]
    v   = g[9:12]
    print("\nASTRA cone_vec geometry:")
    print(f"  Source   : x={src[0]:8.3f}, z={src[1]:8.3f}, y={src[2]:8.3f}")
    print(f"  Detector : x={det[0]:8.3f}, z={det[1]:8.3f}, y={det[2]:8.3f}")
    print(f"  U vector : x={u[0]:8.4f}, z={u[1]:8.4f}, y={u[2]:8.4f}")
    print(f"  V vector : x={v[0]:8.4f}, z={v[1]:8.4f}, y={v[2]:8.4f}")

def unity_geom12_from_world_coords(
    src_world: np.ndarray,     # xraySource.position
    obj_world: np.ndarray,     # imagedObject.position
    det_world: np.ndarray,     # detObject.position
    initial_calibration: np.ndarray, # (3, 3) array of initial offsets to apply to src_world, obj_world, det_world respectively, to match Astra's geometry convention. Each row is a (x,y,z) offset.
    obj_rot_y_deg: float,      # imagedObject.rotation.eulerAngles.y
    alpha: float,               # initial angle offset (degrees) to apply to obj_rot_y_deg, if needed to match Astra's angle=0 convention
    astra_scaling: float,      # astra scaling factor to convert from Unity units to mm (e.g. 1.0 if 1 unit = 1 mm)f
    det_spacing: float,        # astraDetectorSpacing
    det_col: np.ndarray,        # xraySource.up
    det_row: np.ndarray,     # xraySource.right
    offset_x: float = 0.0,          # Optional additional offset in x (Unity right) direction, applied after rotation
    offset_z: float = 0.0           # Optional additional offset in z (Unity forward) direction, applied after rotation
) -> np.ndarray:
    obj_rot = np.deg2rad(float(obj_rot_y_deg))
    alpha = np.deg2rad(float(alpha))
    offset_x_rot = offset_x * np.cos(obj_rot) - offset_z * np.sin(obj_rot)
    offset_z_rot = offset_x * np.sin(obj_rot) + offset_z * np.cos(obj_rot)

    src_world = np.asarray(src_world, dtype=np.float32) + initial_calibration[0]
    obj_world = np.asarray(obj_world, dtype=np.float32) + initial_calibration[1] + np.array([offset_x_rot, 0.0, offset_z_rot], dtype=np.float32)
    det_world = np.asarray(det_world, dtype=np.float32) + initial_calibration[2]
    det_col = np.asarray(det_col, dtype=np.float32)
    det_row = np.asarray(det_row, dtype=np.float32)

    # srcPos / detPos in "object space" (relative to object) and then scaled by SDD
    srcPos = (src_world - obj_world) * astra_scaling
    detPos = (det_world - obj_world) * astra_scaling
    srcPos[1] = -srcPos[1]
    detPos[1] = -detPos[1]

    # u and v from source basis
    u = det_col * det_spacing
    v = det_row * det_spacing

    # Rotate around Y by object rotation
    cosY = float(np.cos(obj_rot + alpha))
    sinY = float(np.sin(obj_rot + alpha))

    srcPos = rotate_y_xz(srcPos, cosY, sinY)
    detPos = rotate_y_xz(detPos, cosY, sinY)
    u = rotate_y_xz(u, cosY, sinY)
    v = rotate_y_xz(v, cosY, sinY)

    geom12 = np.concatenate([pack_xzy(srcPos), pack_xzy(detPos), pack_xzy(u), pack_xzy(v)]).astype(np.float32)
    return geom12


def fetch_and_save_projections(out_dir: str, src_world: np.ndarray, obj_world: np.ndarray, det_world_base: np.ndarray, 
                                alpha: float, angles_deg: np.ndarray, offset_x: float, offset_z: float,
                                image_height: int, image_width: int, initial_calibration: np.ndarray,
                                astra_scaling: float, det_spacing: float, voxel_size: float,
                                det_col: np.ndarray, det_row: np.ndarray, filename_prefix: str = "proj", 
                                phantom_name: str = "cuboid_phantom.npy", debug=True):
    reset_folder(out_dir)
    rec = np.load(phantom_name)

    if debug:
        write_scan_settings_txt(
            out_dir=out_dir,
            image_width=image_width,
            image_height=image_height,
            voxel_size=voxel_size,
            det_spacing=det_spacing,
            src_world=src_world,
            obj_world=obj_world,
            det_world=det_world_base,
            initial_calibration=initial_calibration,
            astra_scaling=astra_scaling,
            angles_deg=angles_deg,
        )

    server = AstraServer(object=rec, image_width=image_width, image_height=image_height, voxel_size=voxel_size)
    geom12_array = []

    # 1) Build all geometries first
    for ry in angles_deg:
        geom12 = unity_geom12_from_world_coords(
            src_world=src_world,
            obj_world=obj_world,
            det_world=det_world_base,
            initial_calibration=initial_calibration,
            obj_rot_y_deg=float(ry),
            alpha=float(alpha),
            astra_scaling=astra_scaling,
            det_spacing=det_spacing,
            det_col=det_col,
            det_row=det_row,
            offset_x=offset_x,
            offset_z=offset_z
        )
        geom12_array.append(geom12)

    geom12_array = np.asarray(geom12_array, dtype=np.float32)  # (N, 12)
    
    if debug:
        sod, sdd, mag, inc_deg = geom12_metrics(geom12_array[0])
        print(
            f"view {0:3d} | SOD={sod:8.3f} mm | "
            f"SDD={sdd:8.3f} mm | M={mag:6.3f} | "
            f"incident={inc_deg:6.4f}°"
        )
        np.set_printoptions(suppress=True)

        print(geom12_array[0].reshape(4, 3))
    # 2) Generate all projections in one call (server.generate_images from earlier)
    imgs = server.generate_stacked_images(geom12_array)  # (N, H, W, 3)

    # 3) Post-process + save per angle
    for idx in range(imgs.shape[0]):
        img = imgs[idx]
        img = apply_napari_contrast_and_gamma(
            img, low_percentile=99.5, high_percentile=100.0, gamma=0.2
        )
        Image.fromarray(img).save(os.path.join(out_dir, f"{filename_prefix}_{idx:03d}.png"))
    server.close()
    # print(f"Saved {len(obj_rot_y_degs)} images to: {out_dir}")


if __name__ == "__main__":
    # Must match server.py
    IMAGE_W = 956
    IMAGE_H = 760

    # Must match Unity settings
    astra_scaling = 1
    DET_SPACING = 0.149600

    # ---- Unity world coordinates you mentioned ----
    # source default position
    SRC_WORLD = np.array([ 4.999512, 29.994888,  0.      ], dtype=np.float32)
    # object position
    OBJ_WORLD = np.array([  0.540527, 20 , 600.      ], dtype=np.float32)

    # detector position (world)
    DET_WORLD = np.array([ -25.31836 ,   18.676949, 1059.      ], dtype=np.float32)
    VOXEL_SIZE = 0.1

    # xraySource orientation (world). Use your real values if different.
    # If your source GameObject has default rotation, these are usually:
    DET_ROW = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    DET_COL = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    OUT_DIR = f"fake_projections/test"
    PHANTOM_NAME = f"phantoms/scan2_160x240x498_transposed_rotY180.npy"  # Must be a 3D numpy array representing the phantom volume. Adjust path if needed.

    # Number of angles: we emulate imagedObject.rotation.eulerAngles.y
    N_ANGLES = 360
    obj_rot_y_degs = np.linspace(0.0, 360.0, N_ANGLES, endpoint=False, dtype=np.float32)

    # initial_calibration = np.array([
    #     np.array([0.0, 0.0, 0.0], dtype=np.float32),
    #     np.array([0.0, 0.0, 0.0], dtype=np.float32),
    #     np.array([0.0, 0.0, 0.0], dtype=np.float32),
    # ])
    
    initial_calibration = np.array([
        np.array([ 0.0     , 0.00, 00.00000], dtype=np.float32),
        np.array([-0.540527, 0.00, 14.91920], dtype=np.float32),
        np.array([25.318359, 6.31, 40.00080], dtype=np.float32)
    ])
    geom = {
            'name': 'Scan1',
            'src':np.array([-10.000488,  29.997368,   0.      ], dtype=np.float32),
            'det':np.array([-20.002441,  33.6557  , 959.      ], dtype=np.float32),
            'obj': np.array([  0.540527, 25 , 600.      ], dtype=np.float32),
            'initial_angle_deg': -1.039974,
            'projections': 1434,
            'image_width': 956,
            'image_height': 760,
            'det_spacing': 0.149600,
    }

    fetch_and_save_projections(
        out_dir=OUT_DIR,
        src_world=geom['src'],
        obj_world=geom['obj'],
        det_world_base=geom['det'],
        alpha= 0.0, angles_deg=obj_rot_y_degs, offset_x=0, offset_z=0,
        image_height=IMAGE_H,
        image_width=IMAGE_W,
        initial_calibration=initial_calibration,
        astra_scaling=astra_scaling,
        det_spacing=DET_SPACING,
        voxel_size=VOXEL_SIZE,
        det_col=DET_COL,
        det_row=DET_ROW,
        filename_prefix="proj",
        phantom_name=PHANTOM_NAME
    )
