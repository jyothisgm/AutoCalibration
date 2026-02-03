import socket
import time

import astra
import numpy as np


class AstraServer:
    def __init__(self, object: np.ndarray, image_width=256, image_height=256, voxel_size=1.0):
        n_slices, n_cols, n_rows = object.shape
        min_x = -n_cols / 2 * voxel_size
        max_x = n_cols / 2 * voxel_size
        min_y = -n_rows/ 2 * voxel_size
        max_y = n_rows / 2 * voxel_size
        min_z = -n_slices/ 2 * voxel_size
        max_z = n_slices/ 2 * voxel_size

        self.vol_geom = astra.create_vol_geom(n_cols, n_rows, n_slices, min_x, max_x, min_y, max_y, min_z, max_z)
        self.vol_id = astra.data3d.create('-vol', self.vol_geom, object)
        self.image_width = image_width
        self.image_height = image_height

        dummy_proj_geom = astra.create_proj_geom('parallel3d', 1, 1, image_width, image_height, [0])
        self.proj_id = astra.data3d.create('-sino', dummy_proj_geom)

        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['VolumeDataId'] = self.vol_id
        cfg['ProjectionDataId'] = self.proj_id
        self.alg_id = astra.algorithm.create(cfg)

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
    rec = np.load('cuboid_phantom.npy')
    values, counts = np.unique(rec, return_counts=True)
    print("Value distribution:")
    for v, c in zip(values, counts):
        print(f"  value={v:.6f} : voxels={c}")
    print("Shape of phantom", values.shape)

    server = AstraServer(object=rec, voxel_size=0.1)
    server.run()
