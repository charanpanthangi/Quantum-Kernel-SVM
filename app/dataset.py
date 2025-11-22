"""Dataset utilities for non-linear classification demos.

The goal of this file is to generate simple 2D datasets that look like moons
or circles. These shapes cannot be separated with a straight line, which is
why kernels (classical or quantum) are helpful. Keeping the generation code
in one place makes it easy to swap or tweak datasets later.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.datasets import make_moons, make_circles


def generate_dataset(
    name: str = "moons",
    n_samples: int = 200,
    noise: float = 0.15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a toy dataset for binary classification.

    Args:
        name: Either ``"moons"`` or ``"circles"``.
        n_samples: Number of points to create.
        noise: Random jitter applied to the shapes to make them less tidy.
        random_state: Seed for reproducibility.

    Returns:
        A tuple ``(X, y)`` where ``X`` has shape ``(n_samples, 2)`` and ``y``
        contains class labels ``{0, 1}``.
    """

    name = name.lower()
    if name not in {"moons", "circles"}:
        raise ValueError("Dataset must be either 'moons' or 'circles'.")

    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    else:
        # Circles come as two concentric rings. The inner ring represents one
        # class, the outer ring another. A straight line cannot split rings,
        # which is why kernel methods are needed.
        X, y = make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=0.5,
            random_state=random_state,
        )

    return X, y
