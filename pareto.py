#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm

# ---- Load data ----
# If from file:
df = pd.read_csv("theta_hat.csv")

# If from string (paste your CSV into csv_text):
# from io import StringIO
# df = pd.read_csv(StringIO(csv_text))

# ---- Compute mean(|sum|) across G0..G4 ----
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
cbar.set_label("Mean |sum|")
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

plt.xlabel("Number of Angles (N_ANGLES)")
plt.ylabel("K (number of beads)")
plt.title("Heatmap with Fixed Error Ranges (log-binned)")

# ---- Annotate cells ----
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.iloc[i, j]
        if not np.isnan(val):
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- metrics already computed ----
sum_cols = ["G0_sum", "G1_sum", "G2_sum", "G3_sum", "G4_sum"]
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
plt.figure(figsize=(9, 6))

# all points
plt.scatter(df["mean_it"], df["mean_abs_sum"], alpha=0.4)

# pareto front
plt.plot(front["mean_it"], front["mean_abs_sum"], linewidth=2)
plt.scatter(front["mean_it"], front["mean_abs_sum"], s=70)

# ---- Annotate Pareto points with (K, N) ----
for _, r in front.iterrows():
    label = f"K={r['K']}, N={r['N_ANGLES']}"
    plt.annotate(
        label,
        (r["mean_it"], r["mean_abs_sum"]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=9
    )

plt.xlabel("Mean iterations (across G0–G4)")
plt.ylabel("Mean |sum| (across G0–G4)")
plt.title("Pareto Front: Mean Iterations vs Mean |sum|")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- metrics ----
sum_cols = ["G0_sum", "G1_sum", "G2_sum", "G3_sum", "G4_sum"]
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
plt.figure(figsize=(9, 6))

# all points
plt.scatter(df["mean_it"], df["mean_abs_sum"], alpha=0.35)

# pareto front
plt.plot(front["mean_it"], front["mean_abs_sum"], linewidth=2)
plt.scatter(front["mean_it"], front["mean_abs_sum"], s=80)

# ---- Log scale on Y ----
plt.yscale("log")

# ---- Vertical labels for Pareto points ----
for _, r in front.iterrows():
    label = f"K={int(r['K'])}\nN={int(r['N_ANGLES'])}"
    plt.annotate(
        label,
        (r["mean_it"], r["mean_abs_sum"]),
        textcoords="offset points",
        xytext=(0, 8),          # small vertical offset
        ha="center",
        va="bottom",
        rotation=90,            # vertical text
        fontsize=9
    )

plt.xlabel("Mean iterations (across G0–G4)")
plt.ylabel("Mean |sum| (log scale)")
plt.title("Pareto Front: Mean Iterations vs Mean |sum|")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# %%
