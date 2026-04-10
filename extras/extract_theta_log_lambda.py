from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

THETA_NAMES = ["dSx", "dSy", "dOx", "dOy", "dOz", "dDx", "dDy", "dDz", "alpha", "offset_x", "offset_z"]

RE_HEADER = re.compile(r"#\s*lambda=(\S+)\s+cuboid=(\S+)\s+scenario=(\S+)\s+N_ANGLES=(\d+),\s*K=(\d+)")
RE_VALUES = re.compile(r"^(?!#)([0-9eE+.\-]+(?:\s+[0-9eE+.\-]+)+)\s*$", re.MULTILINE)
RE_SUM    = re.compile(r"#\s*sum\s*=\s*([0-9eE+.\-]+)")

THETA_LOG_DIR = Path(__file__).resolve().parent.parent / "simulated" / "theta_log_lambda"


def parse_theta_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n{2,}", text.strip())

    rows = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        header_m = RE_HEADER.search(block)
        if not header_m:
            continue

        lambda_  = header_m.group(1)
        cuboid   = header_m.group(2)
        scenario = header_m.group(3)
        n_angles = int(header_m.group(4))
        k        = int(header_m.group(5))

        value_lines = RE_VALUES.findall(block)
        sum_m       = RE_SUM.search(block)

        row: dict = {
            "file":     path.name,
            "lambda":   lambda_,
            "cuboid":   cuboid,
            "scenario": scenario,
            "K":        k,
            "N":        n_angles,
        }

        if len(value_lines) >= 1:
            fake_vals = [float(v) for v in value_lines[0].split()]
            for j, name in enumerate(THETA_NAMES):
                row[f"fake_{name}"] = fake_vals[j] if j < len(fake_vals) else None

        if len(value_lines) >= 2:
            hat_vals = [float(v) for v in value_lines[1].split()]
            for j, name in enumerate(THETA_NAMES):
                row[f"hat_{name}"] = hat_vals[j] if j < len(hat_vals) else None

        if len(value_lines) >= 3:
            diff_vals = [float(v) for v in value_lines[2].split()]
            for j, name in enumerate(THETA_NAMES):
                row[f"diff_{name}"] = diff_vals[j] if j < len(diff_vals) else None

        diff_cols = [row.get(f"diff_{name}") for name in THETA_NAMES]
        if any(v is not None for v in diff_cols):
            row["sum"] = sum(abs(v) for v in diff_cols if v is not None)
        elif sum_m:
            row["sum"] = float(sum_m.group(1))

        rows.append(row)

    return rows


def parse_all(folder: Path = THETA_LOG_DIR) -> pd.DataFrame:
    all_rows = []
    for path in sorted(folder.glob("theta_hat_*.txt")):
        try:
            all_rows.extend(parse_theta_file(path))
        except Exception as e:
            print(f"ERROR parsing {path.name}: {e}")

    df = pd.DataFrame(all_rows)

    def scen_num(s):
        m = re.search(r"\d+", str(s))
        return int(m.group()) if m else 9999

    if not df.empty:
        df["_scen_num"] = df["scenario"].apply(scen_num)
        df = (df.sort_values(["lambda", "cuboid", "K", "N", "_scen_num"])
                .drop(columns="_scen_num")
                .reset_index(drop=True))

        priority = ["lambda", "cuboid", "K", "N", "scenario"]
        rest = [c for c in df.columns if c not in priority]
        df = df[priority + rest]

    return df


EXPECTED_LAMBDAS = ["GN", "LM_high", "LM_low"]
EXPECTED_CUBOIDS = ["normal"]
EXPECTED_K       = [3]
EXPECTED_N       = [3, 5, 6, 9, 10, 12, 15, 18, 24, 36, 60, 90, 180, 360]
EXPECTED_SC      = ["G0", "G1", "G2", "G3", "G4"]


def find_missing(df: pd.DataFrame) -> pd.DataFrame:
    import itertools
    expected = pd.DataFrame(
        list(itertools.product(EXPECTED_LAMBDAS, EXPECTED_CUBOIDS, EXPECTED_K, EXPECTED_N, EXPECTED_SC)),
        columns=["lambda", "cuboid", "K", "N", "scenario"],
    )
    found = df[["lambda", "cuboid", "K", "N", "scenario"]].drop_duplicates()
    missing = expected.merge(found, on=["lambda", "cuboid", "K", "N", "scenario"], how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    missing = (missing.sort_values(["lambda", "cuboid", "N", "K", "scenario"])
                      .reset_index(drop=True)[["lambda", "cuboid", "N", "K", "scenario"]])
    return missing


BRACKETS       = [0, 0.1, 1, 3, 10, float("inf")]
BRACKET_LABELS = ["0–0.1", "0.1–1", "1–3", "3–10", "10+"]
BRACKET_COLORS = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]


def plot_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    lambdas = [l for l in EXPECTED_LAMBDAS if l in df["lambda"].unique()]

    cmap = mcolors.ListedColormap(BRACKET_COLORS)
    norm = mcolors.BoundaryNorm(
        boundaries=[-0.5 + i for i in range(len(BRACKET_COLORS) + 1)],
        ncolors=len(BRACKET_COLORS),
    )

    def to_bracket(val):
        if pd.isna(val):
            return float("nan")
        for i in range(len(BRACKETS) - 1):
            if BRACKETS[i] <= val < BRACKETS[i + 1]:
                return i
        return len(BRACKETS) - 2

    # Single heatmap: rows = lambda types, columns = N
    pivot = (
        df.assign(abs_sum=df["sum"].abs())
          .groupby(["lambda", "N"])["abs_sum"]
          .mean()
          .unstack("N")
          .reindex(index=lambdas, columns=EXPECTED_N)
    )

    bracket_grid = pivot.map(to_bracket).values.astype(float)

    fig, ax = plt.subplots(figsize=(14, max(3, len(lambdas) * 0.7 + 1.5)))

    ax.imshow(bracket_grid, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(EXPECTED_N)))
    ax.set_xticklabels(EXPECTED_N, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(lambdas)))
    ax.set_yticklabels(lambdas, fontsize=9)
    ax.set_xlabel("N (number of angles)", fontsize=10)
    ax.set_ylabel("Lambda type", fontsize=10)

    for ri, lam in enumerate(lambdas):
        for ci, n in enumerate(EXPECTED_N):
            val = pivot.loc[lam, n] if (lam in pivot.index and n in pivot.columns) else float("nan")
            if not pd.isna(val):
                b_idx = int(to_bracket(val))
                txt_color = "white" if b_idx in (0, 4) else "black"
                ax.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                        fontsize=6.5, color=txt_color)

    patches = [mpatches.Patch(color=BRACKET_COLORS[i], label=BRACKET_LABELS[i])
               for i in range(len(BRACKET_LABELS))]
    ax.legend(handles=patches, title="mean |sum|", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=8, title_fontsize=8, framealpha=0.9)

    ax.set_title("Mean |sum| by lambda type vs N  (K=3, averaged over scenarios)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap: {out_path}")


if __name__ == "__main__":
    df = parse_all()
    print(f"Parsed {len(df)} rows from {df['file'].nunique() if not df.empty else 0} files")
    if not df.empty:
        print(df[["lambda", "cuboid", "scenario", "N", "K", "sum"]].to_string(index=False))

    out = THETA_LOG_DIR.parent / "theta_log_lambda_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # --- Missing combinations ---
    missing = find_missing(df)
    total_expected = len(EXPECTED_LAMBDAS) * len(EXPECTED_CUBOIDS) * len(EXPECTED_K) * len(EXPECTED_N) * len(EXPECTED_SC)
    print(f"\n{'='*50}")
    print(f"Expected: {total_expected}  |  Found: {len(df)}  |  Missing: {len(missing)}")
    if missing.empty:
        print("All combinations complete.")
    else:
        print(missing.to_string(index=False))
        out_missing = THETA_LOG_DIR.parent / "theta_log_lambda_missing.csv"
        missing.to_csv(out_missing, index=False)
        print(f"\nSaved missing list: {out_missing}")

    # --- Heatmap ---
    if not df.empty:
        out_heatmap = THETA_LOG_DIR.parent / "theta_log_lambda_heatmap.png"
        plot_heatmap(df, out_heatmap)
