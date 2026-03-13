import tensorflow as tf
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from runspecs import trainspecs

dirname = pathlib.Path(__file__).parent
dataset_dir = dirname / "dataset_stencil"
plot_dir = dataset_dir / "plots"
plot_dir.mkdir(exist_ok=True)

print("\n=== LOADING DATASET ===\n")
ds = tf.data.Dataset.load(str(dataset_dir))
print("Dataset loaded from:", dataset_dir)

# --- Extract all samples ---
features_list = []
targets_list = []

print("\n=== EXTRACTING SAMPLES ===\n")
for f, t in ds:
    features_list.append(f.numpy())
    targets_list.append(t.numpy())

features = np.asarray(features_list)
targets = np.asarray(targets_list)

print("Raw shapes:")
print("  features:", features.shape)  # expected (N, F)
print("  targets :", targets.shape)   # expected (N, 1) or (N,)

# --- Flatten robustly ---
N = features.shape[0]
F = features.shape[-1]

features_flat = features.reshape(N, F)

# targets can be (N,), (N,1), or (N,1,1) depending on pipeline; force to (N,)
targets_flat = targets.reshape(N, -1)
if targets_flat.shape[1] != 1:
    raise RuntimeError(f"Expected targets to have 1 column after reshape, got {targets_flat.shape}")
targets_flat = targets_flat[:, 0]

# --- Build DataFrame ---
cols = trainspecs["features"]
if len(cols) != F:
    raise RuntimeError(
        f"Feature count mismatch: trainspecs has {len(cols)} columns, dataset has {F}.\n"
        f"trainspecs features: {cols}"
    )

df = pd.DataFrame(features_flat, columns=cols)
df["WI"] = targets_flat.astype(float)

# --- Save CSV ---
csv_path = dataset_dir / "final_nn_dataset.csv"
df.to_csv(csv_path, index=False)
print("\n=== SAVED CSV TO ===")
print(csv_path)

# --- Radius inspection ---
print("\n=== RADIUS INSPECTION ===")
if "radius" not in df.columns:
    raise RuntimeError("No 'radius' column in dataset. Check trainspecs['features'].")

unique_radii = np.sort(df["radius"].unique())
print(f"Number of unique radii in dataset: {len(unique_radii)}")
print("Unique radii [m]:")
for i, r in enumerate(unique_radii):
    print(f"  {i:2d}: {r:.6f}")

# counts per radius (VERY useful)
counts = df["radius"].value_counts().sort_index()
print("\nCounts per radius:")
print(counts)

if len(unique_radii) > 1:
    dr = np.diff(unique_radii)
    print("\nRadius spacing (delta r):")
    print(dr)

print("\nRadius min / max:")
print("  min radius:", float(unique_radii.min()))
print("  max radius:", float(unique_radii.max()))

# --- Plot helpers ---
def save_scatter_with_radius_lines(df, ycol, outpath, title=None, s=4, alpha=0.25):
    if ycol not in df.columns:
        print(f"Skip (missing column): {ycol}")
        return

    ur = np.sort(df["radius"].unique())
    plt.figure(figsize=(9, 5))
    plt.scatter(df["radius"], df[ycol], s=s, alpha=alpha)

    for r in ur:
        plt.axvline(r, alpha=0.12, linewidth=0.8)

    plt.xlabel("Radius [m]")
    plt.ylabel(ycol)
    plt.title(title or f"{ycol} vs radius")
    plt.grid(True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", outpath)


def save_summary_band(df, ycol, outpath, title=None):
    """Median + p10/p90 per radius. Great for checking trends."""
    if ycol not in df.columns:
        print(f"Skip (missing column): {ycol}")
        return

    g = df.groupby("radius")[ycol]
    summary = pd.DataFrame({
        "p10": g.quantile(0.10),
        "median": g.quantile(0.50),
        "p90": g.quantile(0.90),
    }).reset_index().sort_values("radius")

    plt.figure(figsize=(9, 5))
    plt.plot(summary["radius"], summary["median"], marker="o")
    plt.fill_between(summary["radius"], summary["p10"], summary["p90"], alpha=0.2)

    plt.xlabel("Radius [m]")
    plt.ylabel(ycol)
    plt.title(title or f"{ycol} vs radius (median + p10/p90)")
    plt.grid(True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", outpath)

    # also store the summary table
    summary_csv = outpath.with_suffix(".csv")
    summary.to_csv(summary_csv, index=False)
    print("Saved:", summary_csv)


# --- What to plot ---
# Always plot WI vs radius
save_scatter_with_radius_lines(df, "WI", plot_dir / "WI_vs_radius_scatter.png",
                               title="WI vs Radius (scatter)")

save_summary_band(df, "WI", plot_dir / "WI_vs_radius_summary.png",
                  title="WI vs Radius (median + p10/p90)")

# Pressure: plot mid + upper/lower if present
for col in ["pressure", "pressure_upper", "pressure_lower"]:
    if col in df.columns:
        save_scatter_with_radius_lines(df, col, plot_dir / f"{col}_vs_radius_scatter.png",
                                       title=f"{col} vs Radius (scatter)")
        save_summary_band(df, col, plot_dir / f"{col}_vs_radius_summary.png",
                          title=f"{col} vs Radius (median + p10/p90)")

# Saturation: plot mid + upper/lower if present
for col in ["saturation", "saturation_upper", "saturation_lower"]:
    if col in df.columns:
        save_scatter_with_radius_lines(df, col, plot_dir / f"{col}_vs_radius_scatter.png",
                                       title=f"{col} vs Radius (scatter)")
        save_summary_band(df, col, plot_dir / f"{col}_vs_radius_summary.png",
                          title=f"{col} vs Radius (median + p10/p90)")

print("\n=== PREVIEW ===")
print(df.head())

print("\n=== SUMMARY ===")
print(df.describe())

print("\nDone. Plots are in:", plot_dir)

###plot pressure vs tid

import matplotlib.pyplot as plt

t = range(len(df))  # radindeks = tid

plt.figure(figsize=(8, 4))
plt.plot(t, df["pressure_upper"], label="upper")
plt.plot(t, df["pressure"], label="mid")
plt.plot(t, df["pressure_lower"], label="lower")
plt.xlabel("Row index (time order)")
plt.ylabel("Pressure")
plt.title("Pressure vs time (upper / mid / lower)")
plt.legend()
plt.grid(True)

outpath = plot_dir / "pressure_vs_time_layers.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")
print("Saved:", outpath)
plt.close()

plt.plot(df["pressure_lower"] - df["pressure"])
plt.title("Pressure difference (lower - mid)")
plt.ylabel("Δp")
plt.xlabel("time index")
plt.show()

outpath = plot_dir / "new.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight")
print("Saved:", outpath)
plt.close()
