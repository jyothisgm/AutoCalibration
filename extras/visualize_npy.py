#%%
import numpy as np
import matplotlib.pyplot as plt

vol = np.load(f"../phantoms/scan2_160x240x498.npy")
vol = vol[:, ::-1, :]
points = np.argwhere(vol >= vol.max() - 8000)

z, y, x = points.T

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(x, y, z, s=1)

# correct origin
# ax.invert_yaxis()
ax.invert_zaxis()

# FIX SKEW
ax.set_box_aspect((vol.shape[2], vol.shape[1], vol.shape[0]))

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.view_init(elev=0, azim=0)

plt.show()

# %%
