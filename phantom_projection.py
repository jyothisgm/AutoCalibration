import os
import socket
import numpy as np
from PIL import Image
from astra_server import AstraServer

import shutil, os

def reset_folder(folder):
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)

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
    # Unity packing used in your C# client: (x, z, y)
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

def unity_geom12_from_worldcoords(
    src_world: np.ndarray,     # xraySource.position
    obj_world: np.ndarray,     # imagedObject.position
    det_world: np.ndarray,     # detObject.position
    obj_rot_y_deg: float,      # imagedObject.rotation.eulerAngles.y
    astra_sdd: float,          # astraSourceDetectorDistance
    det_spacing: float,        # astraDetectorSpacing
    src_up: np.ndarray,        # xraySource.up
    src_right: np.ndarray,     # xraySource.right
) -> np.ndarray:
    src_world = np.asarray(src_world, dtype=np.float32)
    obj_world = np.asarray(obj_world, dtype=np.float32)
    det_world = np.asarray(det_world, dtype=np.float32)
    src_up = np.asarray(src_up, dtype=np.float32)
    src_right = np.asarray(src_right, dtype=np.float32)

    # srcPos / detPos in "object space" (relative to object) and then scaled by SDD
    srcPos = (src_world - obj_world) * astra_sdd
    detPos = (det_world - obj_world) * astra_sdd

    # u and v from source basis
    u = src_up * det_spacing
    v = src_right * det_spacing

    # Rotate around Y by object rotation
    angleY = np.deg2rad(float(obj_rot_y_deg))
    cosY = float(np.cos(angleY))
    sinY = float(np.sin(angleY))

    srcPos = rotate_y_xz(srcPos, cosY, sinY)
    detPos = rotate_y_xz(detPos, cosY, sinY)
    u = rotate_y_xz(u, cosY, sinY)
    v = rotate_y_xz(v, cosY, sinY)

    geom12 = np.concatenate([pack_xzy(srcPos), pack_xzy(detPos), pack_xzy(u), pack_xzy(v)]).astype(np.float32)
    return geom12


def pad_volume_to_square_xy(volume, pad_value=0.0):
    Nz, Ny, Nx = volume.shape

    target = max(Nx, Ny)

    pad_x_total = target - Nx
    pad_y_total = target - Ny

    pad_x0 = pad_x_total // 2
    pad_x1 = pad_x_total - pad_x0

    pad_y0 = pad_y_total // 2
    pad_y1 = pad_y_total - pad_y0

    padded = np.pad(
        volume,
        pad_width=((0, 0), (pad_y0, pad_y1), (pad_x0, pad_x1)),
        mode="constant",
        constant_values=pad_value
    )

    return padded


def fetch_and_save_projections(out_dir: str, src_world: np.ndarray, obj_world: np.ndarray,
                               det_world_base: np.ndarray, obj_rot_y_degs: np.ndarray,   # (N,) in degrees
                               image_height: int, image_width: int,
                               astra_sdd: float, det_spacing: float, voxel_size: float,
                               src_up: np.ndarray, src_right: np.ndarray, filename_prefix: str = "proj", phantom_name: str = "cuboid_phantom.npy"):
    reset_folder(out_dir)
    obj_rot_y_degs = np.asarray(obj_rot_y_degs, dtype=np.float32)
    bytes_per_img = image_height * image_width * 3

    rec = np.load(phantom_name)
    rec = pad_volume_to_square_xy(rec, pad_value=0.0)


    server = AstraServer(object=rec, image_width=image_width, image_height=image_height, voxel_size=voxel_size)
    # print_unity_geometry(src_world, obj_world, det_world_base, obj_rot_y_degs[0])

    for idx, ry in enumerate(obj_rot_y_degs):
        geom12 = unity_geom12_from_worldcoords(src_world=src_world, obj_world=obj_world, det_world=det_world_base,
                                               obj_rot_y_deg=float(ry), astra_sdd=astra_sdd, det_spacing=det_spacing,
                                               src_up=src_up, src_right=src_right)
        # np.set_printoptions(precision=3, suppress=True)
        img = server.generate_image(geom12)
        Image.fromarray(img).save(os.path.join(out_dir, f"{filename_prefix}_{int(ry):03d}_{idx:03d}.png"))
    server.close()
    # print(f"Saved {len(obj_rot_y_degs)} images to: {out_dir}")


if __name__ == "__main__":
    # Must match server.py
    IMAGE_W = 2048
    IMAGE_H = 2048

    # Must match Unity settings
    ASTRA_SDD = 1
    DET_SPACING = 0.075

    # ---- Unity world coordinates you mentioned ----
    # source default position
    SRC_WORLD = np.array([0.0, 45.0, 0.0], dtype=np.float32)

    # object position
    OBJ_WORLD = np.array([0.0, 30.0, 570.0], dtype=np.float32)

    # detector position (world)
    DET_WORLD = np.array([0.0, 0.0, 1004.0], dtype=np.float32)
    VOXEL_SIZE = 0.1

    # xraySource orientation (world). Use your real values if different.
    # If your source GameObject has default rotation, these are usually:
    SRC_RIGHT = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    SRC_UP = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Number of angles: we emulate imagedObject.rotation.eulerAngles.y
    N_ANGLES = 36
    obj_rot_y_degs = np.linspace(0.0, 360.0, N_ANGLES, endpoint=False, dtype=np.float32)

    fetch_and_save_projections(
        out_dir="projections_png_real_test",
        src_world=SRC_WORLD,
        obj_world=OBJ_WORLD,
        det_world_base=DET_WORLD,
        obj_rot_y_degs=obj_rot_y_degs,
        image_height=IMAGE_H,
        image_width=IMAGE_W,
        astra_sdd=ASTRA_SDD,
        det_spacing=DET_SPACING,
        voxel_size=VOXEL_SIZE,
        src_up=SRC_UP,
        src_right=SRC_RIGHT,
        filename_prefix="proj",
    )
