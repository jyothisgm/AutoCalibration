from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

THETA_NAMES = ["dSx", "dSy", "dOx", "dOy", "dOz", "dDx", "dDy", "dDz", "alpha", "offset_x", "offset_z"]

RESULTS_COVERAGE_DIR = Path(__file__).resolve().parent.parent / "results" / "cuboid"

# Must match gauss_newton.py LAMBDA_VALUES
LAMBDA_VALUES = [
    {"name": "GN",        "lam": 0.0},
    {"name": "LM_low",    "lam": 1e-4},
    {"name": "LM_normal", "lam": 1e-2},
    {"name": "LM_high",   "lam": 1.0},
]
LAMBDA_ORDER = [lv["name"] for lv in LAMBDA_VALUES]
_LAM_MAP: dict[str, str] = {f"{lv['lam']:.3e}": lv["name"] for lv in LAMBDA_VALUES}

# Must match gauss_newton.py CUBOID_SIZES (order used for heatmap rows)
CUBOID_ORDER = ["coplanar", "wide", "compact", "tall", "square", "small", "normal"]


def _lam_to_name(lam_val: float) -> str:
    key = f"{lam_val:.3e}"
    if key in _LAM_MAP:
        return _LAM_MAP[key]
    return min(LAMBDA_VALUES, key=lambda lv: abs(lv["lam"] - lam_val))["name"]


# ----------------------------
# Regex helpers
# ----------------------------

RE_SCENARIO = re.compile(
    r"Running cuboid=(\S+)\s+scenario=(\S+)\s+N_ANGLES=(\d+),\s*K=(\d+)"
)

RE_ITER00_LAMBDA = re.compile(
    r"^iter 00\b.*lambda=([0-9eE+.\-]+)",
    re.MULTILINE,
)

RE_ITER_HEADER = re.compile(r"^Iteration\s+(\d+)\s*$", re.MULTILINE)

RE_ITER_LINE = re.compile(
    r"^iter\s+(\d+)\s+cost=([0-9eE+.\-]+)\s*->\s*([0-9eE+.\-]+)\s+\|dtheta\|=([0-9eE+.\-]+)\s+lambda=([0-9eE+.\-]+)",
    re.MULTILINE,
)

RE_FAKE_THETA  = re.compile(r"Fake theta:\s*\[([^\]]+)\]", re.DOTALL)
RE_FINAL_THETA = re.compile(r"Final estimated theta:\s*\[([^\]]+)\]", re.DOTALL)
RE_DIFF        = re.compile(r"Diff from Expected:\s*\[([^\]]+)\]", re.DOTALL)


def _parse_array(s: str) -> list[float]:
    return [float(v) for v in s.replace("\n", " ").split()]


def parse_log_file(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="ignore")

    m = RE_SCENARIO.search(text)
    if not m:
        return None

    m_lam = RE_ITER00_LAMBDA.search(text)
    lambda_name = _lam_to_name(float(m_lam.group(1))) if m_lam else "unknown"

    row: dict = {
        "lambda":   lambda_name,
        "file":     path.name,
        "cuboid":   m.group(1),
        "scenario": m.group(2),
        "N":        int(m.group(3)),
        "K":        int(m.group(4)),
    }

    iter_headers = [int(h.group(1)) for h in RE_ITER_HEADER.finditer(text)]
    row["total_iters"] = iter_headers[-1] if iter_headers else None

    iter_lines = list(RE_ITER_LINE.finditer(text))
    if iter_lines:
        first_m, last_m = iter_lines[0], iter_lines[-1]
        ci  = float(first_m.group(2))
        cf  = float(last_m.group(2))
        di  = float(first_m.group(4))
        df_ = float(last_m.group(4))
        row["cost_initial"]      = ci
        row["cost_final"]        = cf
        row["cost_change_pct"]   = 100.0 * (ci - cf) / ci if ci != 0 else None
        row["ddtheta_initial"]   = di
        row["dtheta"]            = df_
        row["dtheta_change_pct"] = 100.0 * (di - df_) / di if di != 0 else None
        row["final_lambda"]      = float(last_m.group(5))

    m_fake = RE_FAKE_THETA.search(text)
    if m_fake:
        vals = _parse_array(m_fake.group(1))
        for j, name in enumerate(THETA_NAMES):
            row[f"fake_{name}"] = vals[j] if j < len(vals) else None

    m_hat = RE_FINAL_THETA.search(text)
    if m_hat:
        vals = _parse_array(m_hat.group(1))
        for j, name in enumerate(THETA_NAMES):
            row[f"hat_{name}"] = vals[j] if j < len(vals) else None

    m_diff = RE_DIFF.search(text)
    if m_diff:
        vals = _parse_array(m_diff.group(1))
        for j, name in enumerate(THETA_NAMES):
            row[f"diff_{name}"] = vals[j] if j < len(vals) else None
        row["sum"] = sum(abs(v) for v in vals)

    return row


def parse_all(folder: Path = RESULTS_COVERAGE_DIR) -> pd.DataFrame:
    best: dict[tuple, tuple[tuple, dict]] = {}

    for path in sorted(folder.rglob("*.log")):
        try:
            row = parse_log_file(path)
        except Exception as e:
            print(f"ERROR parsing {path}: {e}")
            continue
        if row is None:
            continue

        key = (row["lambda"], row["cuboid"], row["scenario"], row["N"], row["K"])
        sort_key = (path.stat().st_mtime, path.name)
        if key not in best or sort_key > best[key][0]:
            best[key] = (sort_key, row)

    all_rows = [v for _, v in best.values()]
    df = pd.DataFrame(all_rows)

    if df.empty:
        return df

    # Drop incomplete runs
    if "sum" in df.columns:
        df = df[df["sum"].notna()].reset_index(drop=True)

    def scen_num(s):
        mm = re.search(r"\d+", str(s))
        return int(mm.group()) if mm else 9999

    lam_order     = {name: i for i, name in enumerate(LAMBDA_ORDER)}
    cuboid_order  = {name: i for i, name in enumerate(CUBOID_ORDER)}
    df["_lam_ord"]    = df["lambda"].map(lambda x: lam_order.get(x, 999))
    df["_cuboid_ord"] = df["cuboid"].map(lambda x: cuboid_order.get(x, 999))
    df["_scen_num"]   = df["scenario"].apply(scen_num)
    df = (df.sort_values(["_lam_ord", "_cuboid_ord", "N", "_scen_num"])
            .drop(columns=["_lam_ord", "_cuboid_ord", "_scen_num"])
            .reset_index(drop=True))

    df["M"] = df["N"] * (2 * df["K"] + df["K"] * (df["K"] - 1) // 2)

    priority = [
        "lambda", "cuboid", "K", "N", "M", "scenario", "file", "sum",
        "total_iters", "cost_initial", "cost_final", "cost_change_pct",
        "ddtheta_initial", "dtheta", "dtheta_change_pct", "final_lambda",
    ]
    rest = [c for c in df.columns if c not in priority]
    df = df[priority + rest]

    return df


# ----------------------------
# Heatmap: rows = cuboid (reversed), cols = N, one panel per lambda type
# ----------------------------

BRACKETS       = [0, 0.2, 0.5, 1, 3, 10, float("inf")]
BRACKET_LABELS = ["0–0.2", "0.2–0.5", "0.5–1", "1–3", "3–10", "10+"]
BRACKET_COLORS = ["#1a9641", "#74c476", "#a6d96a", "#ffffbf", "#fdae61", "#f46d43", "#d7191c"]

EXPECTED_N = [3, 5, 6, 9, 10, 12, 15, 18, 24, 36, 60, 90, 180, 360]
EXPECTED_K = [3]
EXPECTED_SCENARIOS = ["G0", "G1", "G2", "G3", "G4"]


def plot_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    lambdas = [name for name in LAMBDA_ORDER if name in df["lambda"].unique()]
    cuboids = [c for c in reversed(CUBOID_ORDER) if c in df["cuboid"].unique()]
    n_cols  = [n for n in EXPECTED_N if n in df["N"].unique()]

    cmap = mcolors.ListedColormap(BRACKET_COLORS)
    norm = mcolors.BoundaryNorm(
        boundaries=[-0.5 + i for i in range(len(BRACKET_COLORS) + 1)],
        ncolors=len(BRACKET_COLORS),
    )

    def to_bracket(val):
        if pd.isna(val):
            return float("nan")
        for i in range(len(BRACKETS) - 1):
            if BRACKETS[i] <= abs(val) < BRACKETS[i + 1]:
                return i
        return len(BRACKETS) - 2

    fig, axes = plt.subplots(
        len(lambdas), 1,
        figsize=(max(12, len(n_cols) * 0.9 + 2), len(lambdas) * max(3, len(cuboids) * 0.8 + 1.5)),
        squeeze=False,
    )

    patches = [mpatches.Patch(color=BRACKET_COLORS[i], label=BRACKET_LABELS[i])
                for i in range(len(BRACKET_LABELS))]

    for panel_i, lam in enumerate(lambdas):
        ax  = axes[panel_i][0]
        sub = df[df["lambda"] == lam]

        pivot_sum = (
            sub.groupby(["cuboid", "N"])["sum"]
                .mean()
                .unstack("N")
                .reindex(index=cuboids, columns=n_cols)
        )
        pivot_iters = (
            sub.groupby(["cuboid", "N"])["total_iters"]
                .mean()
                .unstack("N")
                .reindex(index=cuboids, columns=n_cols)
        )

        bracket_grid = pivot_sum.map(to_bracket).values.astype(float)

        ax.imshow(bracket_grid, aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(range(len(n_cols)))
        ax.set_xticklabels(n_cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(cuboids)))
        ax.set_yticklabels(cuboids, fontsize=8)
        ax.set_ylabel("Phantom", fontsize=9)
        ax.set_title(f"Method: Gauss Newton without Levenberg-Marquardt", fontsize=10, fontweight="bold")

        for ri, cub in enumerate(cuboids):
            for ci, n in enumerate(n_cols):
                val   = pivot_sum.loc[cub, n]   if (cub in pivot_sum.index   and n in pivot_sum.columns)   else float("nan")
                iters = pivot_iters.loc[cub, n]  if (cub in pivot_iters.index and n in pivot_iters.columns) else float("nan")
                if not pd.isna(val):
                    b_idx = int(to_bracket(val))
                    txt_color = "white" if b_idx in (0, 4) else "black"
                    iters_str = f"\n({int(round(iters))})" if not pd.isna(iters) else ""
                    ax.text(ci, ri, f"{val:.3f}{iters_str}", ha="center", va="center",
                            fontsize=6, color=txt_color, linespacing=1.4)

        if panel_i == len(lambdas) - 1:
            ax.set_xlabel(r"Number of Projections ($N$)", fontsize=9)

    fig.legend(handles=patches, title="MAE", bbox_to_anchor=(1.01, 0.98),
                loc="upper left", fontsize=8, title_fontsize=8, framealpha=0.9)
    fig.suptitle(r"MAE — Spread of 3 beaded phantom vs No of Projections ($N$) (Averaged over 5 scenarios)", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap: {out_path}")


def missing_combinations(df: pd.DataFrame) -> pd.DataFrame:
    import itertools
    full = pd.DataFrame(
        list(itertools.product(CUBOID_ORDER, EXPECTED_K, EXPECTED_N, EXPECTED_SCENARIOS)),
        columns=["cuboid", "K", "N", "scenario"],
    )
    full["M"] = full["N"] * (2 * full["K"] + full["K"] * (full["K"] - 1) // 2)

    if df.empty:
        return full.sort_values(["cuboid", "K", "N", "scenario"]).reset_index(drop=True)

    present = df[["cuboid", "K", "N", "scenario"]].drop_duplicates()
    merged = full.merge(present, on=["cuboid", "K", "N", "scenario"], how="left", indicator=True)
    missing = (merged[merged["_merge"] == "left_only"]
               .drop(columns="_merge")
               .sort_values(["cuboid", "K", "N", "scenario"])
               .reset_index(drop=True))
    return missing


if __name__ == "__main__":
    df = parse_all()
    print(f"Parsed {len(df)} rows  |  cuboids: {sorted(df['cuboid'].unique()) if not df.empty else []}")

    if not df.empty:
        print(df[["lambda", "cuboid", "K", "N", "M", "scenario", "sum", "total_iters"]].to_string(index=False))

    out_csv = RESULTS_COVERAGE_DIR / "coverage_results_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv}")

    missing = missing_combinations(df)
    out_missing = RESULTS_COVERAGE_DIR / "coverage_missing_combinations.csv"
    missing.to_csv(out_missing, index=False)
    print(f"Missing combinations: {len(missing)}  |  Saved: {out_missing}")

    if not df.empty:
        out_heatmap = RESULTS_COVERAGE_DIR / "coverage_results_heatmap.png"
        plot_heatmap(df, out_heatmap)
