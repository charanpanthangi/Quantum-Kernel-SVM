"""Tests for the training pipeline.

We verify that the pipeline runs end-to-end on a miniature dataset and
returns accuracy values in the expected range ``[0, 1]``.
"""

import numpy as np

from app.dataset import generate_dataset
from app.train import run_training_pipeline


def test_training_pipeline_returns_accuracies() -> None:
    """Running the pipeline should produce numeric accuracy values."""

    X, y = generate_dataset(name="moons", n_samples=40, noise=0.1, random_state=0)

    result = run_training_pipeline(X, y, test_size=0.2, random_state=0, reps=1)

    assert 0.0 <= result.classical_accuracy <= 1.0
    assert 0.0 <= result.quantum_accuracy <= 1.0
    assert isinstance(result.classical_model, object)
    assert isinstance(result.quantum_model, object)
    assert isinstance(result.quantum_kernel, object)
