# %%
import numpy as np
import pyvista as pv


#%%
# Measurement in mm
CLEARANCE = 1.0
WIDTH, BREADTH, HEIGHT = 80.0, 80.0, 160.0
NO_OF_BEADS = 3
BEAD_RADIUS = 2.5
MARGIN = 6.0
TURNS = 4.0


#%%
def generate_cuboid_spiral_beads(
    a: float,
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
    Create a cuboid phantom (a x b x h) and place k beads along a spiral path near edges.

    Coordinate convention:
        - Cuboid centered at `center` with extents:
            x in [-a/2, a/2], y in [-b/2, b/2], z in [-h/2, h/2]
        - Spiral progresses along z from bottom to top.

    spiral_mode:
        - "edge-loop": beads traverse a rectangular loop (near faces) with continuous "phase"
        - "rounded-rect": similar but corners are rounded by arcs (smoother path)
    """
    assert k >= 1
    rng = np.random.default_rng(seed)

    cx, cy, cz = center
    x_min, x_max = cx - a/2, cx + a/2
    y_min, y_max = cy - b/2, cy + b/2
    z_min, z_max = cz - h/2, cz + h/2

    # Inner rectangle where bead centers live (keep away from faces by margin + radius)
    safe = bead_radius + clearance + margin
    ix_min, ix_max = x_min + safe, x_max - safe
    iy_min, iy_max = y_min + safe, y_max - safe

    if ix_min >= ix_max or iy_min >= iy_max:
        raise ValueError("margin+bead_radius too large for given cuboid dimensions.")

    # Parameter t in [0, 1) for each bead along the loop; z increases with bead index.
    # theta controls which point on the loop.
    i = np.arange(k, dtype=float)

    z_low  = z_min + safe
    z_high = z_max - safe

    if z_low >= z_high:
        raise ValueError("Cuboid height too small for bead_radius + clearance.")

    z = z_low + (i / max(1, (k - 1))) * (z_high - z_low)

    theta = 2.0 * np.pi * turns * (i / k)

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
        per = 2*(w + d)

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
        # Straight lengths (each reduced by 2r)
        w_s = max(0.0, w - 2*r)
        d_s = max(0.0, d - 2*r)
        # Total perimeter: 2*(w_s+d_s) + 2*pi*r
        per = 2*(w_s + d_s) + 2*np.pi*r
        s = phase01 * per

        # Define corner centers:
        # bottom-right, bottom-left, top-left, top-right
        br = (ix_max - r, iy_min + r)
        bl = (ix_min + r, iy_min + r)
        tl = (ix_min + r, iy_max - r)
        tr = (ix_max - r, iy_max - r)

        # Segment order CCW starting at bottom-right straight (towards bottom-left):
        # 1) bottom straight (length w_s)
        if s < w_s:
            x = (ix_max - r) - s
            y = iy_min
            return x, y
        s -= w_s

        # 2) bottom-left corner arc (quarter circle, angle -pi/2 -> -pi)
        arc = 0.5*np.pi*r
        if s < arc:
            ang = -np.pi/2 - (s / r)
            x = bl[0] + r*np.cos(ang)
            y = bl[1] + r*np.sin(ang)
            return x, y
        s -= arc

        # 3) left straight (length d_s)
        if s < d_s:
            x = ix_min
            y = (iy_min + r) + s
            return x, y
        s -= d_s

        # 4) top-left corner arc (angle -pi -> -3pi/2)
        if s < arc:
            ang = -np.pi - (s / r)
            x = tl[0] + r*np.cos(ang)
            y = tl[1] + r*np.sin(ang)
            return x, y
        s -= arc

        # 5) top straight (length w_s)
        if s < w_s:
            x = (ix_min + r) + s
            y = iy_max
            return x, y
        s -= w_s

        # 6) top-right corner arc (angle -3pi/2 -> -2pi)
        if s < arc:
            ang = -3*np.pi/2 - (s / r)
            x = tr[0] + r*np.cos(ang)
            y = tr[1] + r*np.sin(ang)
            return x, y
        s -= arc

        # 7) right straight (length d_s)
        if s < d_s:
            x = ix_max
            y = (iy_max - r) - s
            return x, y
        s -= d_s

        # 8) bottom-right corner arc (angle 0 -> -pi/2)
        # (wrap)
        ang = 0.0 - (s / r)
        x = br[0] + r*np.cos(ang)
        y = br[1] + r*np.sin(ang)
        return x, y

    # Convert theta to phase in [0,1)
    phase = (theta / (2.0*np.pi)) % 1.0

    xy = np.zeros((k, 2), dtype=float)
    for idx in range(k):
        if spiral_mode == "rounded-rect":
            xy[idx] = point_on_rounded_rect_loop(phase[idx], rounded_rect_r)
        else:
            xy[idx] = point_on_rect_loop(phase[idx])

    x = xy[:, 0]
    y = xy[:, 1]

    # Optional positional jitter in x,y too
    if bead_pos_jitter > 0:
        x = x + rng.normal(0.0, bead_pos_jitter, size=k)
        y = y + rng.normal(0.0, bead_pos_jitter, size=k)

    bead_centers = np.column_stack([x, y, z])

    # --- Build meshes ---
    cuboid = pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))

    beads = []
    for c in bead_centers:
        beads.append(pv.Sphere(radius=bead_radius, center=c, theta_resolution=32, phi_resolution=32))
    beads_mesh = beads[0]
    for m in beads[1:]:
        beads_mesh = beads_mesh + m

    phantom_mesh = cuboid + beads_mesh
    return phantom_mesh, cuboid, beads_mesh, bead_centers


# def voxelize_mesh(mesh: pv.PolyData, spacing: float = 1.0, bounds_pad: float = 2.0):
#     """
#     Simple voxelization via pyvista.voxelize (returns an UnstructuredGrid).
#     You can convert it to a binary volume if needed.
#     """
#     b = mesh.bounds
#     bounds = (b[0]-bounds_pad, b[1]+bounds_pad, b[2]-bounds_pad, b[3]+bounds_pad, b[4]-bounds_pad, b[5]+bounds_pad)
#     vox = pv.voxelize(mesh, density=spacing)
#     return vox

def voxelize_mesh(mesh: pv.PolyData, spacing: float = 1.0):
    vox = mesh.voxelize(spacing=spacing)
    return vox


# %%
if __name__ == "__main__":
    # Example parameters (edit as needed)
    phantom_mesh, cuboid_mesh, beads_mesh, centers = generate_cuboid_spiral_beads(
        a=WIDTH, b=BREADTH, h=HEIGHT, k=NO_OF_BEADS,
        bead_radius=BEAD_RADIUS,
        margin=MARGIN,
        clearance=CLEARANCE,
        turns=TURNS,
        spiral_mode="rounded-rect",   # try "edge-loop" too
        rounded_rect_r=10.0,
        bead_theta_jitter=0.00,
        bead_pos_jitter=0.00,
        seed=66
    )

    # Save meshes
    phantom_mesh.save("cuboid_spiral_beads.vtk")
    cuboid_mesh.save("cuboid_only.vtk")
    beads_mesh.save("beads_only.vtk")
    np.save("bead_centers.npy", centers)

    # Optional: voxelize (useful if you want a binary grid like in your lung/bronchi pipeline)
    vox = voxelize_mesh(phantom_mesh, spacing=1.0)
    vox.save("cuboid_spiral_beads_voxels.vtk")

    # Quick visualization
    p = pv.Plotter()
    p.add_mesh(cuboid_mesh, opacity=0.15)
    p.add_mesh(beads_mesh, opacity=1.0)
    p.show()
