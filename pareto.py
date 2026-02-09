#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# ---- Load data ----
# If from file:
df = pd.read_csv("theta_hat.csv")

#%%
# ---- Compute mean absolute error across G0..G4 ----
sum_cols = ["G0_sum", "G1_sum", "G2_sum", "G3_sum", "G4_sum"]

df["mean_abs_sum"] = df[sum_cols].abs().mean(axis=1)

pivot = df.pivot_table(
    index="K",
    columns="N_ANGLES",
    values="mean_abs_sum",
    aggfunc="mean"
)


bounds = np.array([0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 20.0])
norm = BoundaryNorm(boundaries=bounds, ncolors=256, clip=True)

plt.figure(figsize=(14, 6))
im = plt.imshow(
    pivot,
    aspect="auto",
    origin="lower",
    norm=norm,
    cmap="viridis"
)

cbar = plt.colorbar(im)
cbar.set_label("Mean Absolute Error")
cbar.set_ticks(bounds)
cbar.set_ticklabels([f"{b:g}" for b in bounds])

plt.xticks(
    ticks=np.arange(len(pivot.columns)),
    labels=pivot.columns,
    rotation=45
)
plt.yticks(
    ticks=np.arange(len(pivot.index)),
    labels=pivot.index
)

plt.xlabel("Number of Projection Angles")
plt.ylabel("Number of Beads (K)")
plt.title("Heatmap with Fixed Error Ranges")

# ---- Annotate cells ----
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.iloc[i, j]
        if not np.isnan(val):
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.show()

# %%
# ---- metrics ----
it_cols  = ["G0_it",  "G1_it",  "G2_it",  "G3_it",  "G4_it"]

df["mean_abs_sum"] = df[sum_cols].abs().mean(axis=1)
df["mean_it"] = df[it_cols].mean(axis=1)

# ---- Pareto front (minimize both) ----
def pareto_front_2d(df_in, x_col, y_col):
    d = df_in[[x_col, y_col]].copy()
    order = np.lexsort((d[y_col].to_numpy(), d[x_col].to_numpy()))
    d_sorted = d.iloc[order].reset_index()

    best_y = np.inf
    keep = np.zeros(len(d_sorted), dtype=bool)

    for i, row in d_sorted.iterrows():
        if row[y_col] < best_y:
            keep[i] = True
            best_y = row[y_col]

    return df_in.loc[d_sorted.loc[keep, "index"]]

front = pareto_front_2d(df, "mean_it", "mean_abs_sum") \
            .sort_values(["mean_it", "mean_abs_sum"])

# ---- Plot ----
plt.figure(figsize=(10, 6))

# all points colored by K
Ks = np.sort(df["K"].unique())
cmap = plt.get_cmap("tab10", len(Ks))
K_to_color = {k: cmap(i) for i, k in enumerate(Ks)}

for k in Ks:
    d = df[df["K"] == k]
    plt.scatter(
        d["mean_it"],
        d["mean_abs_sum"],
        color=K_to_color[k],
        alpha=0.45,
        s=55,
        label=f"K={k}"
    )

# pareto front (neutral color, on top)
plt.plot(front["mean_it"], front["mean_abs_sum"], color="black", linewidth=2.2, zorder=5)
plt.scatter(front["mean_it"], front["mean_abs_sum"], color="black", s=90, zorder=6, label="Pareto front")

# ---- Log scale on Y ----
plt.yscale("log")

# ---- Vertical labels for Pareto points (stacked for readability) ----
for _, r in front.iterrows():
    label = f"K={int(r['K'])} N={int(r['N_ANGLES'])}"
    plt.annotate(
        label,
        (r["mean_it"], r["mean_abs_sum"]),
        textcoords="offset points",
        xytext=(0, -8),      # negative → downward
        ha="center",
        va="top",            # anchor at top so text goes down
        rotation=90,
        fontsize=9
    )

# ---- X ticks: show all Pareto points ----
x_ticks = np.sort(front["mean_it"].unique())
plt.xticks(x_ticks, [f"{x:.1f}" for x in x_ticks], rotation=45, ha="right")

# For log y: use decade ticks (clean, no overlap)
# ax = plt.gca()
# ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
# ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

plt.xlabel("Mean iterations for 5 variations")
plt.ylabel("Mean Absolute Error (log scale)")
plt.title("Pareto Front: Mean Iterations vs Mean Absolute Error")

# legend (K colors + Pareto)
plt.legend(ncol=2, fontsize=9)

plt.tight_layout()
plt.show()

# %%
# ---- Compute mean(|sum|) across G0..G4 ----
cost_cols = ["G0_cost", "G1_cost", "G2_cost", "G3_cost", "G4_cost"]

df["mean_abs_sum"] = df[cost_cols].abs().mean(axis=1)

pivot = df.pivot_table(
    index="K",
    columns="N_ANGLES",
    values="mean_abs_sum",
    aggfunc="mean"
)


bounds = np.array([0, 1, 2, 5, 20, 50, 100, 2000])
norm = BoundaryNorm(boundaries=bounds, ncolors=256, clip=True)

plt.figure(figsize=(14, 6))
im = plt.imshow(
    pivot,
    aspect="auto",
    origin="lower",
    norm=norm,
    cmap="viridis"
)

cbar = plt.colorbar(im)
cbar.set_label("Mean Absolute Error")
cbar.set_ticks(bounds)
cbar.set_ticklabels([f"{b:g}" for b in bounds])

plt.xticks(
    ticks=np.arange(len(pivot.columns)),
    labels=pivot.columns,
    rotation=45
)
plt.yticks(
    ticks=np.arange(len(pivot.index)),
    labels=pivot.index
)

plt.xlabel("Number of Projection Angles")
plt.ylabel("Number of Beads (K)")
plt.title("Heatmap with Fixed Error Ranges")

# ---- Annotate cells ----
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.iloc[i, j]
        if not np.isnan(val):
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.show()

# %%
# ---- metrics ----
it_cols  = ["G0_it",  "G1_it",  "G2_it",  "G3_it",  "G4_it"]

df["mean_abs_sum"] = df[cost_cols].abs().mean(axis=1)
df["mean_it"] = df[it_cols].mean(axis=1)

# ---- Pareto front (minimize both) ----
def pareto_front_2d(df_in, x_col, y_col):
    d = df_in[[x_col, y_col]].copy()
    order = np.lexsort((d[y_col].to_numpy(), d[x_col].to_numpy()))
    d_sorted = d.iloc[order].reset_index()

    best_y = np.inf
    keep = np.zeros(len(d_sorted), dtype=bool)

    for i, row in d_sorted.iterrows():
        if row[y_col] < best_y:
            keep[i] = True
            best_y = row[y_col]

    return df_in.loc[d_sorted.loc[keep, "index"]]

front = pareto_front_2d(df, "mean_it", "mean_abs_sum") \
            .sort_values(["mean_it", "mean_abs_sum"])

# ---- Plot ----
plt.figure(figsize=(10, 6))

# all points colored by K
Ks = np.sort(df["K"].unique())
cmap = plt.get_cmap("tab10", len(Ks))
K_to_color = {k: cmap(i) for i, k in enumerate(Ks)}

for k in Ks:
    d = df[df["K"] == k]
    plt.scatter(
        d["mean_it"],
        d["mean_abs_sum"],
        color=K_to_color[k],
        alpha=0.45,
        s=55,
        label=f"K={k}"
    )

# pareto front (neutral color, on top)
plt.plot(front["mean_it"], front["mean_abs_sum"], color="black", linewidth=2.2, zorder=5)
plt.scatter(front["mean_it"], front["mean_abs_sum"], color="black", s=90, zorder=6, label="Pareto front")

# ---- Log scale on Y ----
plt.yscale("log")

# ---- Vertical labels for Pareto points (stacked for readability) ----
for _, r in front.iterrows():
    label = f"K={int(r['K'])} N={int(r['N_ANGLES'])}"
    plt.annotate(
        label,
        (r["mean_it"], r["mean_abs_sum"]),
        textcoords="offset points",
        xytext=(0, -8),      # negative → downward
        ha="center",
        va="top",            # anchor at top so text goes down
        rotation=90,
        fontsize=9
    )

# ---- X ticks: show all Pareto points ----
x_ticks = np.sort(front["mean_it"].unique())
plt.xticks(x_ticks, [f"{x:.1f}" for x in x_ticks], rotation=45, ha="right")

# For log y: use decade ticks (clean, no overlap)
# ax = plt.gca()
# ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
# ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

plt.xlabel("Mean iterations for 5 variations")
plt.ylabel("Mean Absolute Error (log scale)")
plt.title("Pareto Front: Mean Iterations vs Mean Absolute Error")

# legend (K colors + Pareto)
plt.legend(ncol=2, fontsize=9)

plt.tight_layout()
plt.show()