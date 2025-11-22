"""Plotting helpers to visualize datasets and decision boundaries.

Plotting makes the abstract kernel concept easier to grasp. The functions
below generate contour plots for the classical RBF SVM and the quantum-kernel
SVM, letting users see how the boundaries wrap around the non-linear shapes.
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from .quantum_kernel import compute_kernel_matrix


def create_mesh_grid(X: np.ndarray, padding: float = 0.5, step: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """Create a 2D grid that covers the data range.

    The grid is used to evaluate model predictions at many points so we can
    plot smooth contours.
    """

    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    return xx, yy


def plot_dataset(X: np.ndarray, y: np.ndarray, path: str) -> None:
    """Save a scatter plot of the raw dataset.

    Args:
        X: Feature array.
        y: Labels.
        path: Output image path.
    """

    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="k")
    plt.title("Toy dataset: moons or circles")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_decision_boundaries(
    classical_model,
    quantum_model,
    quantum_kernel,
    classical_scaler,
    quantum_scaler,
    data_splits: dict,
    output_path: str,
) -> None:
    """Plot side-by-side decision regions for classical and quantum SVMs.

    Args:
        classical_model: Fitted scikit-learn pipeline with scaler + SVC.
        quantum_model: Fitted SVC using a precomputed kernel.
        quantum_kernel: QuantumKernel instance for evaluating new points.
        classical_scaler: Scaler used by the classical pipeline.
        quantum_scaler: Min-max scaler used before quantum feature mapping.
        data_splits: Dictionary containing ``X_train``, ``X_test``, ``y_train``,
            and ``y_test``.
        output_path: Where to save the combined plot.
    """

    X_train = data_splits["X_train"]
    y_train = data_splits["y_train"]
    X_all = np.concatenate([data_splits["X_train"], data_splits["X_test"]])
    y_all = np.concatenate([data_splits["y_train"], data_splits["y_test"]])

    xx, yy = create_mesh_grid(X_all)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Classical predictions: scale then predict directly.
    grid_scaled_classical = classical_scaler.transform(grid_points)
    classical_preds = classical_model.named_steps["svc"].predict(grid_scaled_classical)
    classical_preds = classical_preds.reshape(xx.shape)

    # Quantum predictions: scale with quantum scaler, compute kernel against training data.
    grid_scaled_quantum = quantum_scaler.transform(grid_points)
    X_train_scaled = quantum_scaler.transform(X_train)
    kernel_grid = compute_kernel_matrix(quantum_kernel, grid_scaled_quantum, data_reference=X_train_scaled)
    quantum_preds = quantum_model.predict(kernel_grid).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, preds, title in [
        (axes[0], classical_preds, "Classical SVM (RBF)"),
        (axes[1], quantum_preds, "Quantum Kernel SVM"),
    ]:
        contour = ax.contourf(xx, yy, preds, cmap="coolwarm", alpha=0.3)
        ax.scatter(X_all[:, 0], X_all[:, 1], c=y_all, cmap="viridis", edgecolors="k", s=25)
        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.grid(alpha=0.2)
        fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
