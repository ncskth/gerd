import argparse
import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCALES = ["s0.25", "s0.5", "s1.0", "s2.0", "s4.0"]
SCALE_TO_INDEX = {scale: idx for idx, scale in enumerate(SCALES)}
SCALE_PATTERN = re.compile(r"(s0\.25|s0\.5|s2\.0|s4\.0)")


def detect_scale(name: str):
    """Return one of the known scale tags if present in name, else None."""
    match = SCALE_PATTERN.search(name)
    if match:
        return match.group(1)
    return None


def load_distributions_from_test_dir(test_dir: Path):
    """Load total/per-shape pixel error distributions from a test directory."""
    pkl_path = test_dir / "test_results.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        l2_values = np.asarray(payload.get("l2_distances", []), dtype=float)
        per_shape_payload = payload.get("per_shape_l2", {})
        per_shape_values = {
            shape: np.asarray(per_shape_payload.get(shape, []), dtype=float)
            for shape in ["square"]
        }

        if l2_values.size == 0:
            metrics = payload.get("metrics", {})
            mean = metrics.get("total_l2_mean")
            if mean is None:
                raise ValueError(f"Missing l2_distances and total_l2_mean in {pkl_path}")
            l2_values = np.asarray([float(mean)], dtype=float)
            print(
                f"Warning: Using single-point fallback for boxplot in {pkl_path.name} "
                f"(no l2_distances found)."
            )

        for shape in ["square"]:
            if per_shape_values[shape].size == 0:
                metrics = payload.get("metrics", {})
                shape_mean = metrics.get(f"{shape}_l2_mean")
                if shape_mean is not None:
                    per_shape_values[shape] = np.asarray([float(shape_mean)], dtype=float)

        return l2_values, per_shape_values

    txt_path = test_dir / "test_metrics.txt"
    if txt_path.exists():
        mean = None
        for line in txt_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("L2 Error (mean):"):
                mean = float(line.split(":", 1)[1].replace("px", "").strip())

        if mean is None:
            raise ValueError(f"Could not parse L2 mean from {txt_path}")

        print(
            f"Warning: {txt_path.name} only has summary stats; "
            f"using single-point fallback for boxplot."
        )
        empty_per_shape = {
            "circle": np.asarray([], dtype=float),
            "triangle": np.asarray([], dtype=float),
            "square": np.asarray([], dtype=float),
        }
        return np.asarray([mean], dtype=float), empty_per_shape

    raise FileNotFoundError(
        f"No test metrics found in {test_dir} (expected test_results.pkl or test_metrics.txt)"
    )


def collect_experiment_data(exp_dir: Path):
    """Collect scale -> (l2_values, per_shape_values, test_dir) in one experiment.
    Also include test-s1.0 if it exists."""
    test_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("test-")]
    test_dirs = sorted(test_dirs, key=lambda p: p.stat().st_mtime)

    scale_data = {}
    for test_dir in test_dirs:
        scale = detect_scale(test_dir.name)
        if scale is None:
            continue
        l2_values, per_shape_values = load_distributions_from_test_dir(test_dir)
        # If multiple tests exist for one scale, keep the latest by mtime.
        scale_data[scale] = (l2_values, per_shape_values, test_dir)

    # Also include test-s1.0 if it exists
    test_s1_dir = exp_dir / "test-s1.0"
    if test_s1_dir.exists() and test_s1_dir.is_dir():
        try:
            l2_values, per_shape_values = load_distributions_from_test_dir(test_s1_dir)
            scale_data["s1.0"] = (l2_values, per_shape_values, test_s1_dir)
        except Exception as e:
            print(f"Warning: Could not load test-s1.0: {e}")

    return scale_data


def plot_experiments(experiment_data, output_path: Path):
    """Plot boxplots over scales and connect medians for each experiment."""
    base_x = np.arange(len(SCALES), dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    num_experiments = len(experiment_data)
    if num_experiments == 1:
        offsets = np.array([0.0])
        box_width = 0.45
    else:
        offsets = np.linspace(-0.28/2, 0.28/2, num_experiments)
        box_width = 0.18

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])

    for exp_idx, (exp_name, scale_data) in enumerate(experiment_data.items()):
        positions = []
        box_data = []
        median_x = []
        median_y = []
        color = colors[exp_idx % len(colors)]

        for scale in SCALES:
            if scale not in scale_data:
                continue
            idx = SCALE_TO_INDEX[scale]
            l2_values, _, _ = scale_data[scale]
            if l2_values.size == 0:
                continue
            pos = base_x[idx] + offsets[exp_idx]
            positions.append(pos)
            box_data.append(l2_values)
            median_x.append(pos)
            median_y.append(float(np.median(l2_values)))

        if not box_data:
            continue

        boxplot = ax.boxplot(
            box_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
            patch.set_edgecolor(color)
        for whisker in boxplot["whiskers"]:
            whisker.set_color(color)
        for cap in boxplot["caps"]:
            cap.set_color(color)
        for median in boxplot["medians"]:
            median.set_color(color)
            median.set_linewidth(2.0)

        ax.plot(
            median_x,
            median_y,
            marker="o",
            linewidth=2,
            color=color,
            label=exp_name,
        )

    ax.set_xticks(base_x)
    ax.set_xticklabels(SCALES)
    ax.set_xlabel("Scale")
    ax.set_ylabel("Pixel Error (px)")
    ax.set_title("Test Pixel Error Across Scales (Boxplots + Median)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_shape_experiments(experiment_data, output_path: Path):
    """Plot per-shape boxplots over scales and connect medians.
    Boxplots show Q1, median, Q3, whiskers (1.5*IQR), and far outliers."""
    base_x = np.arange(len(SCALES), dtype=float)
    shapes = ["square"]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharey=True)
    num_experiments = len(experiment_data)

    if num_experiments == 1:
        offsets = np.array([0.0])
        box_width = 0.45
    else:
        offsets = np.linspace(-0.28/2, 0.28/2, num_experiments)
        box_width = 0.18

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])

    for exp_idx, (exp_name, scale_data) in enumerate(experiment_data.items()):
        positions = []
        box_data = []
        median_x = []
        median_y = []
        color = colors[exp_idx % len(colors)]

        for scale in SCALES:
            if scale not in scale_data:
                continue
            idx = SCALE_TO_INDEX[scale]
            _, per_shape_values, _ = scale_data[scale]
            shape_values = np.asarray(per_shape_values.get('square', []), dtype=float)
            if shape_values.size == 0:
                continue

            pos = base_x[idx] + offsets[exp_idx]
            positions.append(pos)
            box_data.append(shape_values)
            median_x.append(pos)
            median_y.append(float(np.median(shape_values)))

        if not box_data:
            continue

        boxplot = ax.boxplot(
            box_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=True,  # Show far outliers
            boxprops=dict(facecolor=color, alpha=0.3),
            medianprops=dict(color=color, linewidth=2),
            whiskerprops=dict(color=color, linestyle='--'),
            capprops=dict(color=color),
            flierprops=dict(marker='o', markerfacecolor=color, markersize=5, alpha=0.5),
        )
        # Connect medians
        ax.plot(median_x, median_y, color=color, marker="o", linewidth=2, label=exp_name)
    ax.set_xticks(base_x)
    ax.set_xticklabels(SCALES)
    ax.set_xlabel("Scale")
    ax.set_ylabel("Pixel Error (px)")
    ax.set_title(f"Square Error (Boxplot)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot test pixel error across scales for 1 to 3 experiment directories."
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        help="One, two, or three experiment directories.",
    )
    args = parser.parse_args()

    if not (1 <= len(args.experiment_dirs) <= 3):
        print("ERROR: Please provide between 1 and 3 experiment directories.")
        sys.exit(1)

    experiment_data = {}

    for exp_arg in args.experiment_dirs:
        exp_dir = Path(exp_arg)
        if not exp_dir.exists() or not exp_dir.is_dir():
            print(f"ERROR: Experiment directory not found: {exp_dir}")
            sys.exit(1)

        scale_data = collect_experiment_data(exp_dir)
        if not scale_data:
            print(
                f"WARNING: No test folders with scale tag found in {exp_dir}. "
                f"Expected folder names containing one of: {', '.join(SCALES)}"
            )

        experiment_data[exp_dir.name] = scale_data

    if all(len(v) == 0 for v in experiment_data.values()):
        print("ERROR: No valid test data found in any provided experiment directory.")
        sys.exit(1)

    figures_dir = Path(__file__).resolve().parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_path = figures_dir / "pixel_error_vs_scale.png"
    plot_experiments(experiment_data, output_path)

    # New: plot per-shape error
    per_shape_output_path = figures_dir / "per_shape_pixel_error_vs_scale.png"
    plot_per_shape_experiments(experiment_data, per_shape_output_path)

    print("\nSaved figures:")
    print(f"  {output_path}")
    print(f"  {per_shape_output_path}")
    print("\nLoaded data:")
    for exp_name, scale_data in experiment_data.items():
        print(f"  {exp_name}")
        for scale in SCALES:
            if scale in scale_data:
                l2_values, _, test_dir = scale_data[scale]
                median = float(np.median(l2_values))
                q1, q3 = np.percentile(l2_values, [25, 75])
                print(
                    f"    {scale}: n={len(l2_values)}, median={median:.3f}px, "
                    f"IQR=[{q1:.3f}, {q3:.3f}]px ({test_dir.name})"
                )
            else:
                print(f"    {scale}: missing")


if __name__ == "__main__":
    main()
