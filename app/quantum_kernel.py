"""Quantum kernel utilities built with Qiskit.

This module shows how to turn classical feature vectors into quantum states
using a *feature map*. Qiskit's :class:`~qiskit.circuit.library.ZZFeatureMap`
uses simple rotations (to create superposition) and controlled rotations
(to create entanglement) so that data points become patterns of phases in a
quantum circuit. The :class:`~qiskit_machine_learning.kernels.QuantumKernel`
class then computes a **kernel matrix**, which measures how similar two sets
of data look after the quantum encoding. Classical SVMs can use that matrix
to draw a non-linear decision boundary without us writing complex math.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel


def create_feature_map(num_features: int, reps: int = 2, entanglement: str = "linear") -> ZZFeatureMap:
    """Create a ZZFeatureMap tailored to the input size.

    Args:
        num_features: Number of dimensions in the classical data. The circuit
            will create one qubit per feature, so two-dimensional moons data
            uses two qubits.
        reps: How many times to repeat the feature map pattern. More repeats
            mean a richer embedding at the cost of longer circuits.
        entanglement: Pattern for connecting qubits (``"linear"`` or ``"full"``).

    Returns:
        A configured :class:`ZZFeatureMap` object.
    """

    # The ZZFeatureMap uses RZ and CX gates internally. RZ gates put each qubit
    # into a phase rotation (a simple form of superposition for real numbers).
    # Controlled rotations (CX/ZZ) create entanglement so that features interact.
    feature_map = ZZFeatureMap(
        feature_dimension=num_features,
        reps=reps,
        entanglement=entanglement,
        data_map_func=lambda x: x,  # keep mapping simple for beginners
    )
    return feature_map


def create_quantum_kernel(
    num_features: int,
    reps: int = 2,
    entanglement: str = "linear",
    seed: int = 42,
    sampler: Optional[Sampler] = None,
) -> QuantumKernel:
    """Build a QuantumKernel instance ready for kernel computations.

    Args:
        num_features: Number of features in the classical data.
        reps: Number of repetitions in the feature map.
        entanglement: Entanglement style passed to ``ZZFeatureMap``.
        seed: Random seed for reproducible simulations.
        sampler: Optional Qiskit sampler. If not provided, a default
            :class:`Sampler` is created.

    Returns:
        A configured :class:`QuantumKernel` object.
    """

    # The random seed keeps simulated results reproducible across runs.
    algorithm_globals.random_seed = seed
    q_sampler = sampler or Sampler()
    feature_map = create_feature_map(num_features=num_features, reps=reps, entanglement=entanglement)

    # QuantumKernel automates the heavy lifting: it transpiles the feature map,
    # executes it on the sampler, and turns overlaps into a kernel matrix.
    quantum_kernel = QuantumKernel(feature_map=feature_map, sampler=q_sampler)
    return quantum_kernel


def compute_kernel_matrix(
    quantum_kernel: QuantumKernel,
    data_primary: np.ndarray,
    data_reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a quantum kernel matrix for one or two datasets.

    Args:
        quantum_kernel: A prepared :class:`QuantumKernel` instance.
        data_primary: Feature array of shape ``(n_samples, n_features)`` that
            forms the rows of the kernel matrix.
        data_reference: Optional second feature array forming the columns of the
            kernel matrix. If omitted, the function computes the self-kernel
            where both axes use ``data_primary``.

    Returns:
        A 2D NumPy array containing similarity scores between samples. Higher
        values mean two points look more alike after quantum encoding.
    """

    # The evaluate method runs the quantum circuit for each pair of samples.
    # When ``data_reference`` is ``None`` it computes all pairwise overlaps of
    # ``data_primary`` with itself.
    kernel = quantum_kernel.evaluate(x_vec=data_primary, y_vec=data_reference)
    # Convert to a dense NumPy array for compatibility with scikit-learn.
    return np.asarray(kernel)
