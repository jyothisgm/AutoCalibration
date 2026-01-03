# client_get_projections_and_save_pngs.py
# Sends N cone_vec geometries to your server.py and saves each returned projection as a PNG.

import os
import socket
import numpy as np
from PIL import Image


def cone_vec_row(
    angle_rad: float,
    DSO: float,
    DOD: float,
    det_spacing_x: float,
    det_spacing_y: float,
    det_offset_u: float = 0.0,
    det_offset_v: float = 0.0,
) -> np.ndarray:
    s = np.sin(angle_rad)
    c = np.cos(angle_rad)

    src = np.array([ s * DSO, -c * DSO, 0.0], dtype=np.float32)
    det = np.array([-s * DOD,  c * DOD, 0.0], dtype=np.float32)

    # u: detector column direction (pixel step in cols)
    u = np.array([c * det_spacing_x, s * det_spacing_x, 0.0], dtype=np.float32)
    # v: detector row direction (pixel step in rows)
    v = np.array([0.0, 0.0, det_spacing_y], dtype=np.float32)

    det = det + det_offset_u * u + det_offset_v * v
    return np.concatenate([src, det, u, v]).astype(np.float32)


def recvall(conn: socket.socket, nbytes: int) -> bytes:
    data = bytearray()
    while len(data) < nbytes:
        chunk = conn.recv(nbytes - len(data))
        if not chunk:
            raise ConnectionError("Server closed connection while receiving.")
        data.extend(chunk)
    return bytes(data)


def fetch_and_save_projections(
    host: str,
    port: int,
    out_dir: str,
    angles_rad: np.ndarray,
    image_height: int,
    image_width: int,
    DSO: float,
    DOD: float,
    det_spacing_x: float,
    det_spacing_y: float,
    det_offset_u: float = 0.0,
    det_offset_v: float = 0.0,
    filename_prefix: str = "proj",
    recv_timeout_s: float = 30.0,
):
    os.makedirs(out_dir, exist_ok=True)

    angles_rad = np.asarray(angles_rad, dtype=np.float32)
    bytes_per_img = image_height * image_width * 3

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(recv_timeout_s)
        s.connect((host, port))

        for idx, ang in enumerate(angles_rad):
            geom12 = cone_vec_row(
                angle_rad=float(ang),
                DSO=DSO,
                DOD=DOD,
                det_spacing_x=det_spacing_x,
                det_spacing_y=det_spacing_y,
                det_offset_u=det_offset_u,
                det_offset_v=det_offset_v,
            )

            # Send 12 float32 = 48 bytes
            s.sendall(geom12.tobytes(order="C"))

            # Receive one RGB uint8 image
            img_bytes = recvall(s, bytes_per_img)
            img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(image_height, image_width, 3)

            # Save as PNG
            im = Image.fromarray(img, mode="RGB")
            im.save(os.path.join(out_dir, f"{filename_prefix}_{idx:04d}.png"))

    print(f"Saved {len(angles_rad)} images to: {out_dir}")


if __name__ == "__main__":
    # Must match server.py detector size expectations
    HOST = "127.0.0.1"
    PORT = 50007
    IMAGE_W = 256
    IMAGE_H = 256

    # Geometry (units must match server vol_geom units)
    DSO = 800.0
    DOD = 400.0
    DET_PX_X = 0.2
    DET_PX_Y = 0.2

    # Angles
    N_ANGLES = 360
    angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False, dtype=np.float32)

    fetch_and_save_projections(
        host=HOST,
        port=PORT,
        out_dir="projections_png",
        angles_rad=angles,
        image_height=IMAGE_H,
        image_width=IMAGE_W,
        DSO=DSO,
        DOD=DOD,
        det_spacing_x=DET_PX_X,
        det_spacing_y=DET_PX_Y,
        det_offset_u=0.0,
        det_offset_v=0.0,
        filename_prefix="proj",
        recv_timeout_s=60.0,
    )
