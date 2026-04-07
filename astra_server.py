import socket
import time

import astra
import numpy as np


class AstraServer:
    def __init__(self, object: np.ndarray, image_width=256, image_height=256, voxel_size=1.0):
        n_slices, n_rows, n_cols = object.shape
        min_x = -n_cols / 2 * voxel_size
        max_x = n_cols / 2 * voxel_size
        min_z = -n_rows/ 2 * voxel_size
        max_z = n_rows / 2 * voxel_size
        min_y = -n_slices/ 2 * voxel_size
        max_y = n_slices/ 2 * voxel_size

        # print(f"Volume geometry: cols={n_cols}, rows={n_rows}, slices={n_slices}")
        # print(f"Volume bounds: x=[{min_x:.2f}, {max_x:.2f}], y=[{min_y:.2f}, {max_y:.2f}], z=[{min_z:.2f}, {max_z:.2f}]")

        # Unity XYZ -> Astra XZY
        self.vol_geom = astra.create_vol_geom(n_rows, n_cols, n_slices, min_x, max_x, min_z, max_z, min_y, max_y)
        self.vol_id = astra.data3d.create('-vol', self.vol_geom, object)
        self.image_width = image_width
        self.image_height = image_height

        dummy_proj_geom = astra.create_proj_geom('parallel3d', 1, 1, image_height, image_width, [0])
        self.proj_id = astra.data3d.create('-sino', dummy_proj_geom)

        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['VolumeDataId'] = self.vol_id
        cfg['ProjectionDataId'] = self.proj_id
        self.alg_id = astra.algorithm.create(cfg)
        # track current number of projections stored in proj_id
        self._current_N = 1

    def _delete_proj_and_alg(self):
        if self.alg_id is not None:
            astra.algorithm.delete(self.alg_id)
            self.alg_id = None
        if self.proj_id is not None:
            astra.data3d.delete(self.proj_id)
            self.proj_id = None

    def close(self):
        # call this on shutdown if you want clean exit
        self._delete_proj_and_alg()
        if self.vol_id is not None:
            astra.data3d.delete(self.vol_id)
            self.vol_id = None
        # optional final clear
        astra.clear()

    def generate_image(self, geometry_vector):
        proj_geom = astra.create_proj_geom('cone_vec', self.image_height, self.image_width, geometry_vector.reshape([1, 12]))
        astra.data3d.change_geometry(self.proj_id, proj_geom)
        astra.algorithm.run(self.alg_id)
        image = astra.data3d.get(self.proj_id).squeeze()
        image = np.clip(image, 0, np.percentile(image, 99.99))
        max_val = image.max()
        # print(f"Generated image max value: {max_val:.4f}")
        if max_val == 0:
            normalized = np.zeros_like(image, dtype=np.uint8)
        else:
            normalized = (255 * image / max_val).astype(np.uint8)
        return np.dstack([normalized, normalized, normalized])

    def generate_stacked_images(self, geometry_vectors, normalize=False):
        gv = np.asarray(geometry_vectors, dtype=np.float32)
        if gv.ndim == 1:
            gv = gv.reshape(1, 12)
        if gv.shape[1] != 12:
            raise ValueError(f"Expected (N,12) or (12,), got {gv.shape}")

        N = gv.shape[0]

        proj_geom = astra.create_proj_geom("cone_vec", self.image_height, self.image_width, gv)
        need_recreate = (getattr(self, "_current_N", None) != N)

        if need_recreate:
            # Clean up old objects (ignore errors if already deleted)
            if hasattr(self, "alg_id") and self.alg_id is not None:
                try:
                    astra.algorithm.delete(self.alg_id)
                except Exception:
                    pass
                self.alg_id = None

            if hasattr(self, "proj_id") and self.proj_id is not None:
                try:
                    astra.data3d.delete(self.proj_id)
                except Exception:
                    pass
                self.proj_id = None

            # Recreate proj data with correct N
            # NOTE: choose the correct data type for your algorithm:
            # - '-sino' is common for projection/sinogram data in ASTRA configs
            # - Some setups use '-proj3d' depending on config style
            self.proj_id = astra.data3d.create("-sino", proj_geom)

            # Recreate the algorithm that uses self.proj_id
            # You must rebuild the config exactly like you originally did.
            cfg = astra.astra_dict("FP3D_CUDA")
            cfg["VolumeDataId"] = self.vol_id
            cfg["ProjectionDataId"] = self.proj_id

            # plus any other fields you set originally (ProjectorId, options, etc.)
            self.alg_id = astra.algorithm.create(cfg)
            self._current_N = N

        else:
            # Safe to just update vectors (N unchanged)
            astra.data3d.change_geometry(self.proj_id, proj_geom)

        # Run once to generate N projections
        astra.algorithm.run(self.alg_id)

        images = astra.data3d.get(self.proj_id)  # typically (N, H, W) or (H, N, W) depending on ASTRA object type
        images = np.squeeze(images)

        # If it came as (H, N, W), transpose to (N, H, W)
        if images.ndim == 3 and images.shape[0] == self.image_height and images.shape[1] == N:
            images = np.transpose(images, (1, 0, 2))
            # print(f"Images shape after transpose: {images.shape}")

        if normalize:
            out = np.empty_like(images, dtype=np.uint8)
            for i in range(images.shape[0]):
                img = np.clip(images[i], 0, np.percentile(images[i], 99.99))
                max_val = img.max()
                if max_val == 0:
                    out[i] = np.zeros_like(img, dtype=np.uint8)
                else:
                    out[i] = (255 * img / max_val).astype(np.uint8)
            return out

        return images

    def run(self, host="127.0.0.1", port=50007):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen(1)
            print(f'Listening on {host}:{port}')
            conn, addr = s.accept()
            with conn:
                print("Connected by", addr)
                while True:
                    geometry_vector_bytes = conn.recv(12 * 4)
                    if not geometry_vector_bytes:
                        print("Client disconnected")
                        break
                    geometry_vector = np.frombuffer(geometry_vector_bytes, dtype=np.float32, count=12)
                    img = self.generate_image(geometry_vector)
                    conn.sendall(img.tobytes())

if __name__ == '__main__':
    rec = np.load(f'phantoms/cuboid_phantom.npy')
    values, counts = np.unique(rec, return_counts=True)
    print("Value distribution:")
    for v, c in zip(values, counts):
        print(f"  value={v:.6f} : voxels={c}")
    print("Shape of phantom", values.shape)

    server = AstraServer(object=rec, voxel_size=0.1)
    server.run()
