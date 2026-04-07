#%%
import numpy as np
import matplotlib.pyplot as plt

vol = np.load(f"../phantoms/scan2_160x240x498_transposed_rotY180_new.npy")
# vol = vol[:, ::-1, :]
points = np.argwhere(vol >= vol.max() - 50000)

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

# ax.view_init(elev=0, azim=0)

plt.show()

# %%
volume = np.load(f"../phantoms/scan2_160x240x498_transposed_rotY180_new.npy")
# volume = volume[:, ::-1, :]
z, y, x = volume.shape
print(f"Volume shape: x={x}, y={y}, z={z}")
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
axis_limits = (0, 600)

plt.imshow(volume[28], cmap="gray")
plt.xlim(axis_limits)
plt.ylim(axis_limits)

plt.title("Axial")

plt.subplot(1,3,2)
plt.imshow(volume[:, 20, :], cmap="gray", origin="upper")
plt.xlim(axis_limits)
plt.ylim(axis_limits)
plt.title("Coronal")

plt.subplot(1,3,3)
plt.imshow(volume[:, :, 30], cmap="gray")
plt.xlim(axis_limits)
plt.ylim(axis_limits)
plt.title("Sagittal")

plt.tight_layout()
plt.show()


# %%
