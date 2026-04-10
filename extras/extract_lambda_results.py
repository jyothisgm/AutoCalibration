from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

THETA_NAMES = ["dSx", "dSy", "dOx", "dOy", "dOz", "dDx", "dDy", "dDz", "alpha", "offset_x", "offset_z"]

RESULTS_LAMBDA_DIR = Path(__file__).resolve().parent.parent / "results" / "K*N*G"

# Must match gauss_newton.py LAMBDA_VALUES
LAMBDA_VALUES = [
    {"name": "GN",         "lam": 0.0},
    {"name": "LM_low",     "lam": 1e-4},
    {"name": "LM_normal",  "lam": 1e-2},
    {"name": "LM_high",    "lam": 1.0},
]
# Ordered for heatmap rows
LAMBDA_ORDER = [lv["name"] for lv in LAMBDA_VALUES]

# Build lookup: round-trip via string repr to match float formatting robustly
_LAM_MAP: dict[str, str] = {}
for _lv in LAMBDA_VALUES:
    _key = f"{_lv['lam']:.3e}"
    _LAM_MAP[_key] = _lv["name"]


def _lam_to_name(lam_val: float) -> str:
    """Map a lambda float (from iter 00) to its LAMBDA_VALUES name."""
    key = f"{lam_val:.3e}"
    if key in _LAM_MAP:
        return _LAM_MAP[key]
    # Fallback: closest match
    best = min(LAMBDA_VALUES, key=lambda lv: abs(lv["lam"] - lam_val))
    return best["name"]


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

RE_FAKE_THETA = re.compile(
    r"Fake theta:\s*\[([^\]]+)\]",
    re.DOTALL,
)

RE_FINAL_THETA = re.compile(
    r"Final estimated theta:\s*\[([^\]]+)\]",
    re.DOTALL,
)

RE_DIFF = re.compile(
    r"Diff from Expected:\s*\[([^\]]+)\]",
    re.DOTALL,
)


def _parse_array(s: str) -> list[float]:
    return [float(v) for v in s.replace("\n", " ").split()]


def parse_log_file(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="ignore")

    m = RE_SCENARIO.search(text)
    if not m:
        return None

    # Identify lambda type from iter 00
    m_lam = RE_ITER00_LAMBDA.search(text)
    if m_lam:
        lambda_name = _lam_to_name(float(m_lam.group(1)))
    else:
        lambda_name = "unknown"

    row: dict = {
        "lambda":   lambda_name,
        "file":     path.name,
        "cuboid":   m.group(1),
        "scenario": m.group(2),
        "N":        int(m.group(3)),
        "K":        int(m.group(4)),
    }

    # Iteration stats
    iter_headers = [int(m.group(1)) for m in RE_ITER_HEADER.finditer(text)]
    row["total_iters"] = iter_headers[-1] if iter_headers else None

    iter_lines = list(RE_ITER_LINE.finditer(text))
    if iter_lines:
        first_m = iter_lines[0]
        last_m  = iter_lines[-1]
        row["cost_initial"]    = float(first_m.group(2))
        row["cost_final"]      = float(last_m.group(2))
        row["ddtheta_initial"] = float(first_m.group(4))
        row["dtheta"]          = float(last_m.group(4))
        row["final_lambda"]    = float(last_m.group(5))
        ci, cf = row["cost_initial"], row["cost_final"]
        row["cost_change_pct"] = 100.0 * (ci - cf) / ci if ci != 0 else None
        di, df_ = row["ddtheta_initial"], row["dtheta"]
        row["dtheta_change_pct"] = 100.0 * (di - df_) / di if di != 0 else None

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


def parse_all(folder: Path = RESULTS_LAMBDA_DIR) -> pd.DataFrame:
    # Collect all rows; track best (latest mtime) per (lambda, cuboid, scenario, N, K).
    # Latest file wins — deduplicates runs of the same type/scenario/N.
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
        # (mtime, name) so ties break deterministically
        sort_key = (path.stat().st_mtime, path.name)
        if key not in best or sort_key > best[key][0]:
            best[key] = (sort_key, row)

    all_rows = [v for _, v in best.values()]
    df = pd.DataFrame(all_rows)

    if df.empty:
        return df

    def scen_num(s):
        mm = re.search(r"\d+", str(s))
        return int(mm.group()) if mm else 9999

    # Sort by LAMBDA_VALUES order, then scenario
    lam_order = {name: i for i, name in enumerate(LAMBDA_ORDER)}
    df["_lam_ord"] = df["lambda"].map(lambda x: lam_order.get(x, 999))
    df["_scen_num"] = df["scenario"].apply(scen_num)
    df = (df.sort_values(["_lam_ord", "cuboid", "K", "N", "_scen_num"])
            .drop(columns=["_lam_ord", "_scen_num"])
            .reset_index(drop=True))

    priority = [
        "lambda", "cuboid", "K", "N", "scenario", "file", "sum",
        "total_iters", "cost_initial", "cost_final", "cost_change_pct",
        "ddtheta_initial", "dtheta", "dtheta_change_pct", "final_lambda",
    ]
    rest = [c for c in df.columns if c not in priority]
    df = df[priority + rest]

    return df


# ----------------------------
# Heatmap
# ----------------------------

BRACKETS       = [0, 0.1, 1, 3, 10, float("inf")]
BRACKET_LABELS = ["0–0.1", "0.1–1", "1–3", "3–10", "10+"]
BRACKET_COLORS = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]

EXPECTED_N = [3, 5, 6, 9, 10, 12, 15, 18, 24, 36, 60, 90, 180, 360]


def plot_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    # Rows in LAMBDA_VALUES order, only those present
    lambdas = [name for name in LAMBDA_ORDER if name in df["lambda"].unique()]

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

    n_cols = [n for n in EXPECTED_N if n in df["N"].unique()]

    pivot_sum = (
        df.groupby(["lambda", "N"])["sum"]
          .mean()
          .unstack("N")
          .reindex(index=lambdas, columns=n_cols)
    )

    pivot_iters = (
        df.groupby(["lambda", "N"])["total_iters"]
          .mean()
          .unstack("N")
          .reindex(index=lambdas, columns=n_cols)
    )

    bracket_grid = pivot_sum.map(to_bracket).values.astype(float)

    fig, ax = plt.subplots(figsize=(max(10, len(n_cols) * 0.9 + 2), max(3, len(lambdas) * 1.2 + 1.5)))

    ax.imshow(bracket_grid, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(n_cols)))
    ax.set_xticklabels(n_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(lambdas)))
    ax.set_yticklabels(lambdas, fontsize=9)
    ax.set_xlabel("N (number of angles)", fontsize=10)
    ax.set_ylabel("Lambda type", fontsize=10)

    for ri, lam in enumerate(lambdas):
        for ci, n in enumerate(n_cols):
            val  = pivot_sum.loc[lam, n]   if (lam in pivot_sum.index   and n in pivot_sum.columns)   else float("nan")
            iters = pivot_iters.loc[lam, n] if (lam in pivot_iters.index and n in pivot_iters.columns) else float("nan")
            if not pd.isna(val):
                b_idx = int(to_bracket(val))
                txt_color = "white" if b_idx in (0, 4) else "black"
                iters_str = f"\n({int(round(iters))})" if not pd.isna(iters) else ""
                ax.text(ci, ri, f"{val:.3f}{iters_str}", ha="center", va="center",
                        fontsize=6.5, color=txt_color, linespacing=1.4)

    patches = [mpatches.Patch(color=BRACKET_COLORS[i], label=BRACKET_LABELS[i])
               for i in range(len(BRACKET_LABELS))]
    ax.legend(handles=patches, title="mean |sum|", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=8, title_fontsize=8, framealpha=0.9)

    ax.set_title("Mean |sum of diff| by lambda type vs N  (averaged over scenarios)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap: {out_path}")


if __name__ == "__main__":
    df = parse_all()
    print(f"Parsed {len(df)} rows from {df['lambda'].nunique() if not df.empty else 0} lambda types")

    if not df.empty:
        print(df[["lambda", "cuboid", "scenario", "N", "K", "sum"]].to_string(index=False))

    out_csv = RESULTS_LAMBDA_DIR / "lambda_results_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv}")

    if not df.empty:
        out_heatmap = RESULTS_LAMBDA_DIR / "lambda_results_heatmap.png"
        plot_heatmap(df, out_heatmap)
