# %%
import numpy as np
import pyvista as pv
from pathlib import Path


# ---- Parameters ----
CLEARANCE = 0.0
WIDTH, BREADTH, HEIGHT = 20.0, 40.0, 60.0
NO_OF_BEADS = 6
BEAD_RADIUS = 2.0
MARGIN = 1.0
TURNS = 1.0
HERE = Path(__file__).resolve().parent

#%%
def generate_cuboid_spiral_beads(
    w: float,
    b: float,
    h: float,
    k: int,
    bead_radius: float = 2.0,
    margin: float = 5.0,
    clearance: float = 1.0,
    turns: float = 3.0,
    start_z: float = 0.0,
    center=(0.0, 0.0, 0.0),
    spiral_mode: str = "edge-loop",  # "edge-loop" or "rounded-rect"
    rounded_rect_r: float = 8.0,     # only used if spiral_mode="rounded-rect"
    bead_theta_jitter: float = 0.0,  # radians
    bead_pos_jitter: float = 0.0,    # same units as a,b,h
    seed: int = 66
):
    """
    Create a cuboid phantom (w x b x h) and place k beads along a spiral path near edges.

    Coordinate convention:
        - Cuboid centered at `center` with extents:
            x in [-w/2, w/2], y in [-b/2, b/2], z in [-h/2, h/2]
        - Spiral progresses along z from bottom to top.

    spiral_mode:
        - "edge-loop": beads traverse a rectangular loop (near faces) with continuous "phase"
        - "rounded-rect": similar but corners are rounded by arcs (smoother path)
    """
    assert k >= 1
    rng = np.random.default_rng(seed)

    cx, cy, cz = center
    x_min, x_max = cx - w/2, cx + w/2
    y_min, y_max = cy - b/2, cy + b/2
    z_min, z_max = cz - h/2, cz + h/2

    # Inner rectangle where bead centers live (keep away from faces by margin + radius)
    safe = bead_radius + clearance + margin
    ix_min, ix_max = x_min + safe, x_max - safe
    iy_min, iy_max = y_min + safe, y_max - safe
    print(f"Inner rectangle for bead centers: x[{ix_min}, {ix_max}], y[{iy_min}, {iy_max}]")

    if ix_min >= ix_max or iy_min >= iy_max:
        raise ValueError("margin+bead_radius too large for given cuboid dimensions.")

    # Parameter t in [0, 1) for each bead along the loop; z increases with bead index.
    # theta controls which point on the loop.
    i = np.arange(k, dtype=float)

    z_low  = z_min + safe
    z_high = z_max - safe
    print(f"Bead z range: [{z_low}, {z_high}]")

    if z_low >= z_high:
        raise ValueError("Cuboid height too small for bead_radius + clearance.")

    z = z_low + (i / max(1, (k - 1))) * (z_high - z_low)
    print(f"Bead z positions (before jitter): {z}")

    theta = 2 * np.pi * turns * (i / max(1, k - 1))

    print(f"Bead thetas (before jitter): {theta}")
    print(f"Bead : {bead_theta_jitter}")

    # Optional jitter
    if bead_theta_jitter > 0:
        theta = theta + rng.normal(0.0, bead_theta_jitter, size=k)
    if bead_pos_jitter > 0:
        z = z + rng.normal(0.0, bead_pos_jitter, size=k)

    def point_on_rect_loop(phase01: float):
        """
        Map phase in [0,1) to a point along a rectangle perimeter, starting at (ix_max, iy_min),
        going CCW: bottom edge -> left -> top -> right.
        """
        w = ix_max - ix_min
        d = iy_max - iy_min
        per = 2 * (w + d)

        s = phase01 * per

        # bottom edge: (ix_max -> ix_min) at y=iy_min, length w
        if s < w:
            x = ix_max - s
            y = iy_min
            return x, y

        s -= w
        # left edge: (iy_min -> iy_max) at x=ix_min, length d
        if s < d:
            x = ix_min
            y = iy_min + s
            return x, y

        s -= d
        # top edge: (ix_min -> ix_max) at y=iy_max, length w
        if s < w:
            x = ix_min + s
            y = iy_max
            return x, y

        s -= w
        # right edge: (iy_max -> iy_min) at x=ix_max, length d
        x = ix_max
        y = iy_max - s
        return x, y

    def point_on_rounded_rect_loop(phase01: float, r: float):
        """
        Rounded-rectangle loop near edges:
        - Straight segments shortened by corner radius r
        - Corners are quarter-circle arcs
        """
        # Clamp radius
        r = float(r)
        r = max(0.0, min(r, (ix_max - ix_min)/2 - 1e-6, (iy_max - iy_min)/2 - 1e-6))

        w = ix_max - ix_min
        d = iy_max - iy_min
        print(f"Rounded rect corner radius: {r}")
        print(f"Rounded rect straight lengths before clamp: w={w}, d={d}")
        # Straight lengths (each reduced by 2r)
        w_s = max(0.0, w)
        d_s = max(0.0, d)
        # Total perimeter: 2*(w_s+d_s) + 2*pi*r
        per = 2*(w_s + d_s) + 2*np.pi*r
        s = phase01 * per
        print(f"Rounded rect perimeter: {per}, s={s}")

        # Define corner centers:
        # bottom-right, bottom-left, top-left, top-right
        br = (ix_max - r, iy_min + r)
        bl = (ix_min + r, iy_min + r)
        tl = (ix_min + r, iy_max - r)
        tr = (ix_max - r, iy_max - r)
        print(f"Corner centers: br={br}, bl={bl}, tl={tl}, tr={tr}")

        # Segment order CCW starting at bottom-right straight (towards bottom-left):
        # 1) bottom straight (length w_s)
        if s < w_s:
            x = (ix_max) - s
            y = iy_min
            print(f"Bottom straight: x={x}, y={y}")
            return x, y
        s -= w_s

        # 2) bottom-left corner arc (quarter circle, angle -pi/2 -> -pi)
        arc = 0.5*np.pi*r
        if s < arc:
            ang = -np.pi/2 - (s / r)
            x = bl[0] + r*np.cos(ang)
            y = bl[1] + r*np.sin(ang)
            print(f"Bottom-left corner arc: ang={ang}, x={x}, y={y}")
            return x, y
        s -= arc

        # 3) left straight (length d_s)
        if s < d_s:
            x = ix_min
            y = (iy_min) + s
            print(f"Left straight: x={x}, y={y}")
            return x, y
        s -= d_s

        # 4) top-left corner arc (angle -pi -> -3pi/2)
        if s < arc:
            ang = -np.pi - (s / r)
            x = tl[0] + r*np.cos(ang)
            y = tl[1] + r*np.sin(ang)
            print(f"Top-left corner arc: ang={ang}, x={x}, y={y}")  
            return x, y
        s -= arc

        # 5) top straight (length w_s)
        if s < w_s:
            x = (ix_min) + s
            y = iy_max
            print(f"Top straight: x={x}, y={y}")
            return x, y
        s -= w_s

        # 6) top-right corner arc (angle -3pi/2 -> -2pi)
        if s < arc:
            ang = -3*np.pi/2 - (s / r)
            x = tr[0] + r*np.cos(ang)
            y = tr[1] + r*np.sin(ang)
            print(f"Top-right corner arc: ang={ang}, x={x}, y={y}")
            return x, y
        s -= arc

        # 7) right straight (length d_s)
        if s < d_s:
            x = ix_max
            y = (iy_max) - s
            print(f"Right straight: x={x}, y={y}")
            return x, y
        s -= d_s

        # 8) bottom-right corner arc (angle 0 -> -pi/2)
        # (wrap)
        ang = 0.0 - (s / r)
        x = br[0] + r*np.cos(ang)
        y = br[1] + r*np.sin(ang)
        print(f"Final corner arc: ang={ang}, x={x}, y={y}")
        return x, y

    # Convert theta to phase in [0,1)
    print(f"Bead thetas (before jitter): {theta}")
    phase = (theta / (2.0*np.pi)) % 1.0

    xy = np.zeros((k, 2), dtype=float)
    print(spiral_mode)
    print(f"Bead phases (before jitter): {phase}")
    for idx in range(k):
        if spiral_mode == "rounded-rect":
            xy[idx] = point_on_rounded_rect_loop(phase[idx], rounded_rect_r)
        else:
            xy[idx] = point_on_rect_loop(phase[idx])
    print(f"Bead x,y positions (before jitter): \n{xy}")

    x = xy[:, 0]
    y = xy[:, 1]

    # Optional positional jitter in x,y too
    if bead_pos_jitter > 0:
        x = x + rng.normal(0.0, bead_pos_jitter, size=k)
        y = y + rng.normal(0.0, bead_pos_jitter, size=k)

    bead_centers = np.column_stack([x, y, z])

    # --- Build meshes ---
    print(f"Generating cuboid with {k} beads:", x_min, x_max, y_min, y_max, z_min, z_max)
    cuboid = pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))

    beads = []
    for c in bead_centers:
        beads.append(pv.Sphere(radius=bead_radius, center=c, theta_resolution=32, phi_resolution=32))
    beads_mesh = beads[0]
    for m in beads[1:]:
        beads_mesh = beads_mesh + m

    phantom_mesh = cuboid + beads_mesh
    return phantom_mesh, cuboid, beads_mesh, bead_centers

def make_cuboid_with_beads_volume(width: float, breadth: float, height: float, bead_centers_mm: np.ndarray, bead_radius_mm: float,
                                  voxel_size_mm: float = 1.0, cuboid_level: float = 0.1, bead_level: float = 1.0):
    """
    Returns a single volume[z, y, x] containing:
      - full cuboid
      - embedded bead spheres
    """
    Nx = int(np.round(width / voxel_size_mm))
    Ny = int(np.round(breadth / voxel_size_mm))
    Nz = int(np.round(height / voxel_size_mm))

    # --- Fill entire cuboid ---
    volume = np.full((Nz, Ny, Nx), cuboid_level, dtype=np.float32)

    # Voxel center coordinates (mm)
    xs = (np.arange(Nx) + 0.5) * voxel_size_mm - width / 2
    ys = (np.arange(Ny) + 0.5) * voxel_size_mm - breadth / 2
    zs = (np.arange(Nz) + 0.5) * voxel_size_mm - height / 2

    r2 = bead_radius_mm ** 2

    # --- Overwrite bead regions ---
    for (cx, cy, cz) in bead_centers_mm:
        x0 = np.searchsorted(xs, cx - bead_radius_mm)
        x1 = np.searchsorted(xs, cx + bead_radius_mm)
        y0 = np.searchsorted(ys, cy - bead_radius_mm)
        y1 = np.searchsorted(ys, cy + bead_radius_mm)
        z0 = np.searchsorted(zs, cz - bead_radius_mm)
        z1 = np.searchsorted(zs, cz + bead_radius_mm)

        x0, x1 = max(0, x0), min(Nx, x1)
        y0, y1 = max(0, y0), min(Ny, y1)
        z0, z1 = max(0, z0), min(Nz, z1)

        X = xs[x0:x1][None, None, :]
        Y = ys[y0:y1][None, :, None]
        Z = zs[z0:z1][:, None, None]

        mask = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= r2
        volume[z0:z1, y0:y1, x0:x1][mask] = bead_level
    return volume

def generate_k_bead_phantom(k=NO_OF_BEADS, plot=True, mat=False):
    phantom_path = HERE / f"cuboid_phantom_{k}.npy"
    if phantom_path.exists():
        return  # or just skip saving

    phantom_mesh, cuboid_mesh, beads_mesh, centers = generate_cuboid_spiral_beads(w=WIDTH, b=BREADTH, h=HEIGHT, k=k, bead_radius=BEAD_RADIUS, 
                                                                                  margin=MARGIN, clearance=CLEARANCE, turns=TURNS, spiral_mode="rounded-rect", 
                                                                                  rounded_rect_r=0.0, seed=66)
    print("\nBead center positions (mm):")
    for i, (x, y, z) in enumerate(centers):
        print(f"  Bead {i:02d}: x = {x:.3f}, y = {y:.3f}, z = {z:.3f}")

    # Optional mesh saves (debug / visualization)
    #phantom_mesh.save(HERE / f"cuboid_spiral_beads_{k}.vtk")
    #np.save(HERE / f"bead_centers_{k}.npy", centers)

    # ---- SINGLE NPY PHANTOM ----
    VOXEL_SIZE = 0.1  # mm

    volume = make_cuboid_with_beads_volume(width=WIDTH, breadth=BREADTH, height=HEIGHT, bead_centers_mm=centers, bead_radius_mm=BEAD_RADIUS,
                                           voxel_size_mm=VOXEL_SIZE, cuboid_level=2.0, bead_level=255.0)

    values, counts = np.unique(volume, return_counts=True)
    print("Value distribution:")
    for v, c in zip(values, counts):
        print(f"  value={v:.6f} : voxels={c}")

    np.save(phantom_path, volume)
    print(f"Saved cuboid_phantom_{k}.npy", volume.shape)
    if plot:
        p = pv.Plotter()
        p.add_mesh(cuboid_mesh, opacity=0.15)
        p.add_mesh(beads_mesh, opacity=1.0)
        p.show()


        # Create PyVista grid
        grid = pv.ImageData()
        grid.dimensions = np.array(volume.shape)[::-1] + 1
        grid.spacing = (VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE)

        p = pv.Plotter()
        p.add_volume(
            volume,
            cmap="gray",
            opacity=[0.0, 0.1, 1.0],  # cuboid faint, beads opaque
            shade=True
        )
        p.show()

    if mat:
        import matplotlib.pyplot as plt

        z, y, x = volume.shape
        print(f"Volume shape: x={x}, y={y}, z={z}")
        plt.figure(figsize=(12,4))

        plt.subplot(1,3,1)
        axis_limits = (0, 600)

        plt.imshow(volume[20], cmap="gray")
        plt.xlim(axis_limits)
        plt.ylim(axis_limits)

        plt.title("Axial")

        plt.subplot(1,3,2)
        plt.imshow(volume[:, 20, :], cmap="gray")
        plt.xlim(axis_limits)
        plt.ylim(axis_limits)
        plt.title("Coronal")

        plt.subplot(1,3,3)
        plt.imshow(volume[:, :, 20], cmap="gray")
        plt.xlim(axis_limits)
        plt.ylim(axis_limits)
        plt.title("Sagittal")

        plt.tight_layout()
        plt.show()


# %%
if __name__ == "__main__":
    # Measurement in mm
    generate_k_bead_phantom(k=NO_OF_BEADS, plot=True, mat=True)
