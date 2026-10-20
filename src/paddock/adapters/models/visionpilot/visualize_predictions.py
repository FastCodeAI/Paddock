#!/usr/bin/env python3
"""
Interactive visualization of VisionPilot trajectory predictions.
Compare before/after range-scaled noise model on actual benchmark samples.

Usage:
  streamlit run src/paddock/adapters/models/visionpilot/visualize_predictions.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


@st.cache_data
def load_benchmark_data(run_path: str):
    """Load samples from a benchmark run."""
    samples_file = Path(run_path) / "samples.jsonl"
    if not samples_file.exists():
        return None

    samples = []
    with open(samples_file) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute trajectory metrics."""
    if len(pred) != len(gt):
        pred = pred[: min(len(pred), len(gt))]
        gt = gt[: min(len(pred), len(gt))]

    # ADE: Average Displacement Error
    errors = np.linalg.norm(pred - gt, axis=1)
    ade = np.mean(errors)

    # FDE at 1s, 2s, 3s (assuming 0.5s per step)
    fde_1s = errors[1] if len(errors) > 1 else errors[-1]
    fde_2s = errors[3] if len(errors) > 3 else errors[-1]
    fde_3s = errors[5] if len(errors) > 5 else errors[-1]

    # Longitudinal vs Lateral error
    lon_error = np.mean(np.abs(pred[:, 0] - gt[:, 0]))
    lat_error = np.mean(np.abs(pred[:, 1] - gt[:, 1]))

    return {
        "ADE": ade,
        "FDE@1s": fde_1s,
        "FDE@2s": fde_2s,
        "FDE@3s": fde_3s,
        "Longitudinal MAE": lon_error,
        "Lateral MAE": lat_error,
        "Max Error": np.max(errors),
        "Min Error": np.min(errors),
    }


def plot_trajectory(ax, pred: np.ndarray, gt: np.ndarray, title: str, sample_id: str):
    """Plot a single trajectory comparison."""
    ax.set_title(f"{title}\n{sample_id[:40]}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Distance Ahead (m)", fontsize=9)
    ax.set_ylabel("Lateral Offset (m)", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Plot ground truth
    ax.plot(gt[:, 0], gt[:, 1], "k-", linewidth=2.5, label="Ground Truth", zorder=5)
    ax.scatter(gt[:, 0], gt[:, 1], c="black", s=40, marker="o", zorder=6, alpha=0.7)

    # Plot prediction
    ax.plot(pred[:, 0], pred[:, 1], "g--", linewidth=2, label="Prediction", zorder=4, alpha=0.8)
    ax.scatter(pred[:, 0], pred[:, 1], c="lime", s=30, marker="^", zorder=6, alpha=0.7)

    # Mark start
    ax.scatter(0, 0, c="red", s=150, marker="*", zorder=8, edgecolors="darkred", linewidths=1)

    # Error vectors
    for i in range(0, len(gt), 2):
        error = np.linalg.norm(gt[i] - pred[i])
        if error > 0.2:
            ax.plot([pred[i, 0], gt[i, 0]], [pred[i, 1], gt[i, 1]], "r--", linewidth=0.8, alpha=0.3, zorder=1)

    # Horizon markers
    for i in [1, 3, 5]:
        if i < len(gt):
            ax.axvline(gt[i, 0], color="blue", linestyle=":", alpha=0.15, linewidth=1)

    # Set limits
    all_points = np.vstack([gt, pred])
    x_min, x_max = all_points[:, 0].min() - 1, all_points[:, 0].max() + 1
    y_min, y_max = all_points[:, 1].min() - 1, all_points[:, 1].max() + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    if title == "Prediction":
        ax.legend(loc="best", fontsize=8)


def main():
    st.set_page_config(page_title="VisionPilot Trajectory Viz", layout="wide")
    st.title("🚗 VisionPilot Prediction Visualizer")
    st.markdown("Analyze trajectory predictions - actual vs predicted paths on real benchmark data")

    # Sidebar: configure runs
    st.sidebar.header("Configuration")

    # Mode selection
    mode = st.sidebar.radio("Select mode:", ["Single Run Analysis", "Compare Two Runs"])

    if mode == "Single Run Analysis":
        st.sidebar.info("📊 Showing predictions vs ground truth for a single benchmark run")
        run_path = st.text_input(
            "Run Path",
            value="/localdisk/.tmp/claude-64145/paddock-runs/nuplan/visionpilot/20261007T232811-ddbaba",
            help="Path to benchmark run with samples.jsonl",
        )

        samples = load_benchmark_data(run_path)
        if samples is None:
            st.error("❌ Could not load benchmark data. Check path.")
            st.info(f"Looking for: {run_path}/samples.jsonl")
            return

        st.sidebar.success(f"✅ Loaded {len(samples)} samples")
        baseline_samples = None
        improved_samples = samples

    else:
        st.sidebar.info("📊 Comparing two benchmark runs")
        col1, col2 = st.sidebar.columns(2)

        with col1:
            baseline_run = st.text_input(
                "Baseline Path",
                value="/users/gbanda/git/Paddock/results/nuplan/visionpilot/20260917T220044-97873a",
                help="Path to baseline run",
            )

        with col2:
            improved_run = st.text_input(
                "Improved Path",
                value="/localdisk/.tmp/claude-64145/paddock-runs/nuplan/visionpilot/20261007T232811-ddbaba",
                help="Path to improved run",
            )

        # Load data
        baseline_samples = load_benchmark_data(baseline_run)
        improved_samples = load_benchmark_data(improved_run)

        if baseline_samples is None or improved_samples is None:
            st.error("❌ Could not load benchmark data. Check paths.")
            st.info(f"Baseline: {baseline_run}")
            st.info(f"Improved: {improved_run}")
            return

        st.sidebar.success(f"✅ Loaded {len(baseline_samples)} baseline samples")
        st.sidebar.success(f"✅ Loaded {len(improved_samples)} improved samples")

    # Sample selector
    st.sidebar.header("Select Samples")
    if baseline_samples is None:
        n_samples = len(improved_samples)
    else:
        n_samples = min(len(baseline_samples), len(improved_samples))

    sample_indices = st.sidebar.multiselect(
        "Pick samples to visualize:",
        options=list(range(n_samples)),
        default=[0, 1, 2] if n_samples >= 3 else list(range(n_samples)),
        help="Select up to 5 samples for detailed comparison",
    )

    if not sample_indices:
        st.warning("Please select at least one sample")
        return

    if len(sample_indices) > 5:
        st.warning("Showing first 5 selected samples")
        sample_indices = sample_indices[:5]

    # Overall metrics summary
    st.header("📊 Performance Summary")

    improved_ades = [s["metrics"]["ade"] for s in improved_samples]
    improved_mean = np.mean(improved_ades)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mean ADE", f"{improved_mean:.3f}m", help="Average Displacement Error")

    with col2:
        st.metric("Samples", f"{len(improved_samples)}")

    with col3:
        st.metric("Horizon", "3.0s", help="Prediction horizon (8 waypoints × 0.5s)")

    if baseline_samples is not None:
        baseline_ades = [s["metrics"]["ade"] for s in baseline_samples]
        baseline_mean = np.mean(baseline_ades)
        improvement_pct = (1 - improved_mean / baseline_mean) * 100

        msg = f"📈 **Improvement: {improvement_pct:.1f}%** "
        msg += f"(Baseline: {baseline_mean:.3f}m → Improved: {improved_mean:.3f}m)"
        st.info(msg)

    # Detailed sample comparisons
    st.header("🔍 Sample-by-Sample Analysis")

    if baseline_samples is None:
        # Single run mode
        for idx, sample_idx in enumerate(sample_indices):
            sample = improved_samples[sample_idx]
            with st.expander(f"Sample {sample_idx + 1}: {sample['split']}", expanded=(idx == 0)):
                pred = np.array(sample["prediction"])
                gt = np.array(sample["ground_truth"])
                metrics = compute_metrics(pred, gt)

                # Metrics
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Performance Metrics")
                    for metric_name, metric_value in metrics.items():
                        st.write(f"**{metric_name}**: {metric_value:.4f}")

                with col2:
                    st.subheader("Sample Info")
                    st.write(f"**Scenario**: {sample['split']}")
                    st.write(f"**Waypoints**: {len(pred)}")
                    st.write(f"**Horizon**: {len(pred) * 0.5:.1f}s")

                # Trajectory plot
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_trajectory(ax, pred, gt, "Prediction vs Ground Truth", sample["sample_id"])
                plt.tight_layout()
                st.pyplot(fig)

                # Error over time
                errors = np.linalg.norm(pred - gt, axis=1)
                fig, ax = plt.subplots(figsize=(10, 4))
                time_steps = np.arange(len(errors)) * 0.5
                ax.plot(
                    time_steps,
                    errors,
                    "b-",
                    linewidth=2.5,
                    marker="o",
                    markersize=8,
                    label="Prediction Error",
                )
                ax.fill_between(time_steps, 0, errors, alpha=0.2, color="blue")
                ax.set_xlabel("Time (seconds)", fontsize=10)
                ax.set_ylabel("Error (meters)", fontsize=10)
                ax.set_title("Error Over Prediction Horizon", fontsize=12, fontweight="bold")
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

    else:
        # Two-run comparison mode
        for idx, sample_idx in enumerate(sample_indices):
            split = baseline_samples[sample_idx]["split"]
            with st.expander(f"Sample {sample_idx + 1}: {split}", expanded=(idx == 0)):
                baseline_sample = baseline_samples[sample_idx]
                improved_sample = improved_samples[sample_idx]

                baseline_pred = np.array(baseline_sample["prediction"])
                baseline_gt = np.array(baseline_sample["ground_truth"])
                improved_pred = np.array(improved_sample["prediction"])
                improved_gt = np.array(improved_sample["ground_truth"])

                baseline_metrics = compute_metrics(baseline_pred, baseline_gt)
                improved_metrics = compute_metrics(improved_pred, improved_gt)

                # Metrics comparison
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Baseline")
                    for metric_name, metric_value in baseline_metrics.items():
                        st.write(f"**{metric_name}**: {metric_value:.4f}")

                with col2:
                    st.subheader("Improved")
                    for metric_name, metric_value in improved_metrics.items():
                        st.write(f"**{metric_name}**: {metric_value:.4f}")

                with col3:
                    st.subheader("Δ (Improvement)")
                    for metric_name in baseline_metrics:
                        delta = baseline_metrics[metric_name] - improved_metrics[metric_name]
                        baseline_val = baseline_metrics[metric_name]
                        pct_change = (delta / baseline_val * 100) if baseline_val != 0 else 0
                        color = "🟢" if delta > 0 else "🔴"
                        st.write(f"{color} **{metric_name}**: {delta:+.4f} ({pct_change:+.1f}%)")

                # Trajectory plots
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                plot_trajectory(
                    axes[0], baseline_pred, baseline_gt, "Baseline (Before)", baseline_sample["sample_id"]
                )
                plot_trajectory(
                    axes[1], improved_pred, improved_gt, "Improved (After)", improved_sample["sample_id"]
                )

                plt.tight_layout()
                st.pyplot(fig)

                # Error distribution
                baseline_errors = np.linalg.norm(baseline_pred - baseline_gt, axis=1)
                improved_errors = np.linalg.norm(improved_pred - improved_gt, axis=1)

                fig, ax = plt.subplots(figsize=(10, 4))
                time_steps = np.arange(len(baseline_errors)) * 0.5
                ax.plot(
                    time_steps, baseline_errors, "r--", linewidth=2, marker="o", label="Baseline", alpha=0.7
                )
                ax.plot(
                    time_steps, improved_errors, "g-", linewidth=2, marker="^", label="Improved", alpha=0.7
                )
                ax.fill_between(
                    time_steps,
                    baseline_errors,
                    improved_errors,
                    where=(baseline_errors > improved_errors),
                    alpha=0.2,
                    color="green",
                    label="Improvement",
                )
                ax.set_xlabel("Time (seconds)", fontsize=10)
                ax.set_ylabel("Error (meters)", fontsize=10)
                ax.set_title("Frame-by-Frame Error Comparison", fontsize=12, fontweight="bold")
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

    # Summary statistics
    st.header("📈 Summary Statistics")

    if baseline_samples is None:
        # Single run stats
        st.subheader("ADE Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(improved_ades, bins=15, alpha=0.7, color="green", edgecolor="black")
        ax.axvline(
            np.mean(improved_ades),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(improved_ades):.3f}m",
        )
        ax.set_xlabel("ADE (meters)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title("ADE Distribution Across All Samples", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        # Comparison stats
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ADE Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(baseline_ades, bins=15, alpha=0.6, label="Baseline", color="red", edgecolor="black")
            ax.hist(improved_ades, bins=15, alpha=0.6, label="Improved", color="green", edgecolor="black")
            ax.set_xlabel("ADE (meters)", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title("ADE Distribution Across All Samples", fontsize=12, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.subheader("Improvement Distribution")
            improvements = np.array(baseline_ades) - np.array(improved_ades)
            improvement_pcts = improvements / np.array(baseline_ades) * 100

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(improvement_pcts, bins=15, alpha=0.7, color="blue", edgecolor="black")
            ax.axvline(
                np.mean(improvement_pcts),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(improvement_pcts):.1f}%",
            )
            ax.set_xlabel("Improvement (%)", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title("ADE Improvement Distribution", fontsize=12, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)


if __name__ == "__main__":
    main()
