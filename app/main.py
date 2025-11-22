"""Command-line entry point for running the full demo pipeline.

Usage (from repository root)::

    python -m app.main --dataset moons

The script will:
1. Generate the requested dataset.
2. Train a classical RBF SVM and a quantum-kernel SVM.
3. Save plots showing the raw data and decision boundaries.
4. Print the two accuracies to the console.

Everything uses small defaults so it runs quickly on a laptop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import generate_dataset
from .plots import plot_dataset, plot_decision_boundaries
from .train import run_training_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""

    parser = argparse.ArgumentParser(description="Quantum Kernel SVM demo")
    parser.add_argument(
        "--dataset",
        type=str,
        default="moons",
        choices=["moons", "circles"],
        help="Which toy dataset to generate.",
    )
    parser.add_argument(
        "--samples", type=int, default=200, help="Number of data points to generate."
    )
    parser.add_argument(
        "--noise", type=float, default=0.15, help="Noise level for the dataset."
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="Repetitions for the quantum feature map (controls circuit depth).",
    )
    parser.add_argument(
        "--entanglement",
        type=str,
        default="linear",
        choices=["linear", "full"],
        help="Entanglement pattern for the quantum feature map.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples",
        help="Where to save generated plots.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full workflow and report results."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = generate_dataset(name=args.dataset, n_samples=args.samples, noise=args.noise)

    training_result = run_training_pipeline(
        X=X,
        y=y,
        reps=args.reps,
        entanglement=args.entanglement,
    )

    dataset_plot_path = output_dir / "sample_moons.svg"
    decision_plot_path = output_dir / "sample_results.svg"

    # Save helpful visuals.
    plot_dataset(X, y, path=str(dataset_plot_path))
    plot_decision_boundaries(
        classical_model=training_result.classical_model,
        quantum_model=training_result.quantum_model,
        quantum_kernel=training_result.quantum_kernel,
        classical_scaler=training_result.classical_scaler,
        quantum_scaler=training_result.quantum_scaler,
        data_splits=training_result.data_splits,
        output_path=str(decision_plot_path),
    )

    print("\n=== Quantum Kernel SVM Demo ===")
    print(f"Dataset: {args.dataset} | Samples: {args.samples} | Noise: {args.noise}")
    print(f"Classical RBF SVM accuracy: {training_result.classical_accuracy:.3f}")
    print(f"Quantum kernel SVM accuracy: {training_result.quantum_accuracy:.3f}")
    print(f"Plots saved to: {dataset_plot_path} and {decision_plot_path}\n")


if __name__ == "__main__":
    main()
