# # client_get_projections_and_save_pngs_unity_geom.py
# # Sends N Unity-style cone_vec geometries (12 float32) to your server.py and saves each returned projection as a PNG.
# #
# # Matches the Unity packing in your AstraClient:
# #   [src.x, src.z, src.y, det.x, det.z, det.y, u.x, u.z, u.y, v.x, v.z, v.y]
# #
# # Assumptions:
# # - object center at origin
# # - source and detector rotate around Y axis (object rotation in Unity)
# # - detector plane faces origin
# # - u = xraySource.up * astraDetectorSpacing
# # - v = xraySource.right * astraDetectorSpacing

# import os
# import socket
# import numpy as np
# from PIL import Image


# def recvall(conn: socket.socket, nbytes: int) -> bytes:
#     data = bytearray()
#     while len(data) < nbytes:
#         chunk = conn.recv(nbytes - len(data))
#         if not chunk:
#             raise ConnectionError("Server closed connection while receiving.")
#         data.extend(chunk)
#     return bytes(data)


# def unity_like_cone_vec_row(
#     angle_rad: float,
#     astra_sdd: float,          # Unity: astraSourceDetectorDistance
#     astra_det_spacing: float,  # Unity: astraDetectorSpacing
# ) -> np.ndarray:
#     """
#     Create cone_vec geometry matching your Unity AstraClient (for a Y-axis orbit).

#     Returns float32[12] in Unity packing order: (x, z, y) for each vec3.
#     """
#     s = np.sin(angle_rad).astype(np.float32)
#     c = np.cos(angle_rad).astype(np.float32)

#     # Orbit around Y:
#     # Unity-style ring around origin. These are "object-space" positions after subtracting imagedObject.position.
#     # We place source and detector opposite each other.
#     src_pos = np.array([ s, 0.0, -c], dtype=np.float32) * astra_sdd
#     det_pos = np.array([-s, 0.0,  c], dtype=np.float32) * astra_sdd

#     # Orientation of the source so it faces origin (like LookAt)
#     forward = (-src_pos).astype(np.float32)
#     forward /= (np.linalg.norm(forward) + 1e-12)

#     # Unity world up
#     world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

#     # Right-handed basis
#     right = np.cross(world_up, forward)
#     right /= (np.linalg.norm(right) + 1e-12)

#     up = np.cross(forward, right)
#     up /= (np.linalg.norm(up) + 1e-12)

#     # Unity uses:
#     #   u = xraySource.up * astraDetectorSpacing
#     #   v = xraySource.right * astraDetectorSpacing
#     u = up * astra_det_spacing
#     v = right * astra_det_spacing

#     # Unity packs vec3 into ASTRA vector as (x, z, y)
#     def pack_xzy(vec3: np.ndarray) -> np.ndarray:
#         return np.array([vec3[0], vec3[2], vec3[1]], dtype=np.float32)

#     src_p = pack_xzy(src_pos)
#     det_p = pack_xzy(det_pos)
#     u_p = pack_xzy(u)
#     v_p = pack_xzy(v)

#     geom12 = np.concatenate([src_p, det_p, u_p, v_p]).astype(np.float32)
#     return geom12


# def fetch_and_save_projections(
#     host: str,
#     port: int,
#     out_dir: str,
#     angles_rad: np.ndarray,
#     image_height: int,
#     image_width: int,
#     astra_sdd: float,
#     astra_det_spacing: float,
#     filename_prefix: str = "proj",
#     recv_timeout_s: float = 30.0,
# ):
#     os.makedirs(out_dir, exist_ok=True)

#     angles_rad = np.asarray(angles_rad, dtype=np.float32)
#     bytes_per_img = image_height * image_width * 3

#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.settimeout(recv_timeout_s)
#         s.connect((host, port))

#         for idx, ang in enumerate(angles_rad):
#             geom12 = unity_like_cone_vec_row(
#                 angle_rad=float(ang),
#                 astra_sdd=astra_sdd,
#                 astra_det_spacing=astra_det_spacing,
#             )

#             # Send 12 float32 = 48 bytes
#             s.sendall(geom12.tobytes(order="C"))

#             # Receive RGB uint8 image
#             img_bytes = recvall(s, bytes_per_img)
#             img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(image_height, image_width, 3)

#             Image.fromarray(img).save(os.path.join(out_dir, f"{filename_prefix}_{idx:04d}.png"))

#     print(f"Saved {len(angles_rad)} images to: {out_dir}")


# if __name__ == "__main__":
#     # Must match server.py detector size expectations
#     HOST = "127.0.0.1"
#     PORT = 50007
#     IMAGE_W = 256
#     IMAGE_H = 256

#     # MUST match Unity values
#     ASTRA_SDD = 1024     # AstraClient.astraSourceDetectorDistance
#     DET_SPACING = 0.15      # AstraClient.astraDetectorSpacing

#     # Number of angles
#     N_ANGLES = 8
#     angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False, dtype=np.float32)

#     fetch_and_save_projections(
#         host=HOST,
#         port=PORT,
#         out_dir="projections_png",
#         angles_rad=angles,
#         image_height=IMAGE_H,
#         image_width=IMAGE_W,
#         astra_sdd=ASTRA_SDD,
#         astra_det_spacing=DET_SPACING,
#         filename_prefix="proj",
#         recv_timeout_s=60.0,
#     )



# client_get_projections_and_save_pngs_unity_worldcoords.py
#
# Builds geometry exactly like the Unity C# code, but in Python.
# You provide Unity world coordinates:
#   - xraySource.position  (src_world)
#   - imagedObject.position (obj_world)
#   - detObject.position   (det_world)
# and imagedObject.rotation.eulerAngles.y (deg) per angle.
#
# Then it computes:
#   srcPos = (src_world - obj_world) * SDD
#   detPos = (det_world - obj_world) * SDD
#   u = xraySource.up    * det_spacing
#   v = xraySource.right * det_spacing
# rotates srcPos, detPos, u, v around Y by angleY (deg -> rad)
# packs as: [src.x, src.z, src.y, det.x, det.z, det.y, u.x, u.z, u.y, v.x, v.z, v.y]
# sends to server.py and saves PNGs.

import os
import socket
import numpy as np
from PIL import Image


def recvall(conn: socket.socket, nbytes: int) -> bytes:
    data = bytearray()
    while len(data) < nbytes:
        chunk = conn.recv(nbytes - len(data))
        if not chunk:
            raise ConnectionError("Server closed connection while receiving.")
        data.extend(chunk)
    return bytes(data)


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
    """
    Exact Python replica of the C# snippet you pasted.
    Returns float32[12] suitable for ASTRA 'cone_vec' (single view).
    """
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


def fetch_and_save_projections(
    host: str,
    port: int,
    out_dir: str,
    # Either give per-view detector/object rotations or constant values.
    src_world: np.ndarray,
    obj_world: np.ndarray,
    det_world_base: np.ndarray,
    obj_rot_y_degs: np.ndarray,   # (N,) in degrees
    image_height: int,
    image_width: int,
    astra_sdd: float,
    det_spacing: float,
    src_up: np.ndarray,
    src_right: np.ndarray,
    filename_prefix: str = "proj",
    recv_timeout_s: float = 60.0,
):
    os.makedirs(out_dir, exist_ok=True)
    obj_rot_y_degs = np.asarray(obj_rot_y_degs, dtype=np.float32)

    bytes_per_img = image_height * image_width * 3

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(recv_timeout_s)
        s.connect((host, port))

        for idx, ry in enumerate(obj_rot_y_degs):
            geom12 = unity_geom12_from_worldcoords(
                src_world=src_world,
                obj_world=obj_world,
                det_world=det_world_base,
                obj_rot_y_deg=float(ry),
                astra_sdd=astra_sdd,
                det_spacing=det_spacing,
                src_up=src_up,
                src_right=src_right,
            )

            s.sendall(geom12.tobytes(order="C"))

            img_bytes = recvall(s, bytes_per_img)
            img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(image_height, image_width, 3)

            Image.fromarray(img).save(os.path.join(out_dir, f"{filename_prefix}_{idx:04d}.png"))

    print(f"Saved {len(obj_rot_y_degs)} images to: {out_dir}")


if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 50007

    # Must match server.py
    IMAGE_W = 256
    IMAGE_H = 256

    # Must match Unity settings
    ASTRA_SDD = 256
    DET_SPACING = 0.2

    # ---- Unity world coordinates you mentioned ----
    # source default position
    SRC_WORLD = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # object position
    OBJ_WORLD = np.array([0.0, 0.0, 400.0], dtype=np.float32)

    # detector position (world)
    DET_WORLD = np.array([0.0, 0.0, 1200.0], dtype=np.float32)

    # xraySource orientation (world). Use your real values if different.
    # If your source GameObject has default rotation, these are usually:
    SRC_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    SRC_RIGHT = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Number of angles: we emulate imagedObject.rotation.eulerAngles.y
    N_ANGLES = 8
    obj_rot_y_degs = np.linspace(0.0, 360.0, N_ANGLES, endpoint=False, dtype=np.float32)

    fetch_and_save_projections(
        host=HOST,
        port=PORT,
        out_dir="projections_png",
        src_world=SRC_WORLD,
        obj_world=OBJ_WORLD,
        det_world_base=DET_WORLD,
        obj_rot_y_degs=obj_rot_y_degs,
        image_height=IMAGE_H,
        image_width=IMAGE_W,
        astra_sdd=ASTRA_SDD,
        det_spacing=DET_SPACING,
        src_up=SRC_UP,
        src_right=SRC_RIGHT,
        filename_prefix="proj",
        recv_timeout_s=60.0,
    )
