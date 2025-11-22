"""Unit tests for quantum kernel utilities.

These tests keep the dataset tiny so they run quickly even on machines
without specialized hardware. The goal is simply to check that the helper
functions return matrices of the expected shape and symmetry.
"""

import numpy as np

from app.quantum_kernel import compute_kernel_matrix, create_quantum_kernel


def test_kernel_matrix_shape_and_symmetry() -> None:
    """Quantum kernel matrix should be square and symmetric for self-kernel."""

    rng = np.random.default_rng(0)
    sample_data = rng.random((3, 2))  # 3 samples, 2 features

    q_kernel = create_quantum_kernel(num_features=2, reps=1)
    kernel = compute_kernel_matrix(q_kernel, sample_data)

    assert kernel.shape == (3, 3)
    assert np.allclose(kernel, kernel.T, atol=1e-8)
