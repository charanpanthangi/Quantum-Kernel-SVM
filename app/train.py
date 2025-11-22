"""Training utilities for classical and quantum SVM classifiers.

The functions in this module keep the training logic small and readable.
We train two models:
1. A classical Support Vector Machine (SVM) with the RBF kernel.
2. A quantum-inspired SVM that uses a quantum kernel matrix computed from
   Qiskit's feature map. The model itself is still a classical SVM, but the
   kernel captures non-linear structure thanks to quantum state overlaps.

Everything is written to be friendly to newcomers, avoiding dense equations
and focusing on the practical steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from .quantum_kernel import compute_kernel_matrix, create_quantum_kernel


@dataclass
class TrainingResult:
    """Container for training outputs.

    Attributes:
        classical_accuracy: Accuracy of the classical RBF SVM on the test split.
        quantum_accuracy: Accuracy of the quantum-kernel SVM on the test split.
        classical_model: Fitted scikit-learn pipeline for the RBF SVM.
        quantum_model: Fitted scikit-learn SVC using the quantum kernel.
        quantum_kernel: Prepared QuantumKernel instance for future evaluations.
        classical_scaler: Feature scaler used in the classical pipeline.
        quantum_scaler: Min-max scaler used for quantum inputs.
        data_splits: Dictionary storing train/test arrays for plotting.
    """

    classical_accuracy: float
    quantum_accuracy: float
    classical_model: Pipeline
    quantum_model: SVC
    quantum_kernel: object
    classical_scaler: StandardScaler
    quantum_scaler: MinMaxScaler
    data_splits: Dict[str, np.ndarray]


def train_classical_svm(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> Tuple[float, Pipeline, StandardScaler]:
    """Train a standard RBF-kernel SVM using scikit-learn.

    Returns accuracy, fitted model, and the scaler for later plotting.
    """

    # Standardize the data so each feature has zero mean and unit variance.
    classical_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", gamma="scale")),
        ]
    )
    classical_pipeline.fit(X_train, y_train)
    predictions = classical_pipeline.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predictions)
    scaler: StandardScaler = classical_pipeline.named_steps["scaler"]
    return accuracy, classical_pipeline, scaler


def train_quantum_svm(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    reps: int = 2,
    entanglement: str = "linear",
) -> Tuple[float, SVC, object, MinMaxScaler]:
    """Train an SVM that uses a quantum kernel matrix.

    The model remains a standard :class:`~sklearn.svm.SVC` but expects a
    precomputed kernel matrix instead of raw features. We scale data into the
    range ``[0, π]`` so that the quantum feature map receives smooth angles.
    """

    quantum_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_scaled = quantum_scaler.fit_transform(X_train)
    X_test_scaled = quantum_scaler.transform(X_test)

    q_kernel = create_quantum_kernel(num_features=X_train.shape[1], reps=reps, entanglement=entanglement)
    kernel_train = compute_kernel_matrix(q_kernel, X_train_scaled)
    kernel_test = compute_kernel_matrix(q_kernel, X_test_scaled, data_reference=X_train_scaled)

    quantum_svm = SVC(kernel="precomputed")
    quantum_svm.fit(kernel_train, y_train)
    predictions = quantum_svm.predict(kernel_test)
    accuracy = metrics.accuracy_score(y_test, predictions)

    return accuracy, quantum_svm, q_kernel, quantum_scaler


def run_training_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
    reps: int = 2,
    entanglement: str = "linear",
) -> TrainingResult:
    """Train both classical and quantum SVMs and collect results.

    Args:
        X: Input features.
        y: Labels.
        test_size: Fraction of samples used for testing.
        random_state: Seed for reproducibility.
        reps: Repetitions for the quantum feature map.
        entanglement: Entanglement pattern for the feature map.

    Returns:
        A :class:`TrainingResult` with models, scalers, accuracies, and splits.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    classical_accuracy, classical_model, classical_scaler = train_classical_svm(
        X_train, X_test, y_train, y_test
    )
    quantum_accuracy, quantum_model, q_kernel, quantum_scaler = train_quantum_svm(
        X_train, X_test, y_train, y_test, reps=reps, entanglement=entanglement
    )

    data_splits = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    return TrainingResult(
        classical_accuracy=classical_accuracy,
        quantum_accuracy=quantum_accuracy,
        classical_model=classical_model,
        quantum_model=quantum_model,
        quantum_kernel=q_kernel,
        classical_scaler=classical_scaler,
        quantum_scaler=quantum_scaler,
        data_splits=data_splits,
    )
