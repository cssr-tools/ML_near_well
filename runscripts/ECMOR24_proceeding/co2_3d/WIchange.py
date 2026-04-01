from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import res2df
import res2df.summary


def load_summary_df(deck_path: Path, column_keys="*") -> pd.DataFrame:
    resdatafiles = res2df.ResdataFiles(str(deck_path))
    return res2df.summary.df(resdatafiles, column_keys=column_keys, time_index="raw")


def compute_time_days(df: pd.DataFrame) -> np.ndarray:
    if "TIME" in df.columns:
        return pd.to_numeric(df["TIME"], errors="coerce").to_numpy(dtype=float)
    if "YEARS" in df.columns:
        years = pd.to_numeric(df["YEARS"], errors="coerce").to_numpy(dtype=float)
        return 365.25 * (years - years[0])
    raise KeyError("Fant verken TIME eller YEARS i summary-data.")


def select_last_days(df: pd.DataFrame, time_col: str, last_days: float) -> pd.DataFrame:
    tmax = float(df[time_col].max())
    return df.loc[df[time_col] >= tmax - last_days].copy()


def sort_key_from_name(name: str) -> tuple[int, str]:
    lower = name.lower()
    if lower == "run_0copy":
        return (0, name)
    if "copy" in lower:
        suffix = lower.split("copy", 1)[1]
        if suffix.isdigit():
            return (int(suffix), name)
    return (10**9, name)


def make_groups(run_names: list[str]) -> tuple[list[int], list[int]]:
    group1, group2 = [], []
    for i, name in enumerate(run_names):
        lower = name.lower()
        if lower == "run_0copy":
            group1.append(i)
        elif "copy" in lower:
            suffix = lower.split("copy", 1)[1]
            if suffix.isdigit():
                val = int(suffix)
                if 2 <= val <= 9:
                    group1.append(i)
                elif 70 <= val <= 970 and val % 100 == 70:
                    group2.append(i)
    return group1, group2


def plot_group(series_by_run, indices, title, outpath, ykey, xmin=None, xmax=None):
    if not indices:
        print(f"Ingen runs funnet for: {title}")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.colormaps.get_cmap("tab10")

    y_all = []

    for j, idx in enumerate(indices):
        run_name, df = series_by_run[idx]
        y = pd.to_numeric(df[ykey], errors="coerce").to_numpy(dtype=float)
        x = pd.to_numeric(df["time_days"], errors="coerce").to_numpy(dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)
        if xmin is not None:
            valid &= x >= xmin
        if xmax is not None:
            valid &= x <= xmax

        if np.any(valid):
            color = cmap(j / max(1, len(indices) - 1)) if len(indices) > 1 else cmap(0.0)
            ax.plot(
                x[valid],
                y[valid],
                linewidth=2.0,
                label=run_name,
                color=color,
            )
            y_all.extend(y[valid])

    ax.set_title(title)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel(ykey)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)

    if y_all:
        ymax = max(y_all)
        ymin_fixed = 148000
        yrange = ymax - ymin_fixed

        if yrange <= 0:
            margin = max(abs(ymax) * 0.02, 1.0)
        else:
            margin = 0.05 * yrange

        ax.set_ylim(ymin_fixed, ymax + margin)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--last-days", type=float, default=10.0)
    parser.add_argument("--summary-key", type=str, default="WPI:INJ0")
    parser.add_argument("--outdir", type=Path, default=Path("plots_wi"))
    args = parser.parse_args()

    decks = sorted(args.root.glob("RUN_0copy*.DATA"), key=lambda p: sort_key_from_name(p.stem))
    if not decks:
        raise SystemExit(f"Fant ingen RUN_0copy*.DATA i {args.root}")

    series_by_run = []

    for deck in decks:
        try:
            df = load_summary_df(deck, column_keys=["YEARS", "TIME", args.summary_key])
            df["time_days"] = compute_time_days(df)
            df = select_last_days(df, "time_days", args.last_days)
            series_by_run.append((deck.stem, df))
            print(f"[OK] {deck.stem}")
        except Exception as exc:
            print(f"[SKIP] {deck.name}: {exc}")

    if not series_by_run:
        raise SystemExit("Ingen runs kunne prosesseres.")

    run_names = [name for name, _ in series_by_run]
    group1, group2 = make_groups(run_names)

    print("\nGruppe 1:", [run_names[i] for i in group1])
    print("Gruppe 2:", [run_names[i] for i in group2])

    plot_group(
        series_by_run,
        group1,
        title=f"{args.summary_key}, dag 30-40: RUN_0copy til RUN_0copy9",
        outpath=args.outdir / "WPI_copy_to_copy9_day30_40.png",
        ykey=args.summary_key,
        xmin=30.0,
        xmax=40.0,
    )

    plot_group(
        series_by_run,
        group2,
        title=f"{args.summary_key}, dag 60-70: RUN_0copy70 til RUN_0copy970",
        outpath=args.outdir / "WPI_copy70_to_copy970_day60_70.png",
        ykey=args.summary_key,
        xmin=60.0,
        xmax=70.0,
    )

if __name__ == "__main__":
    main()