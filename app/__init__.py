"""Top-level package for the Quantum Kernel SVM demo.

This module exposes convenience imports so the rest of the codebase can
use short paths like ``from app import dataset``. The project is written
with beginners in mind, so every file contains inline explanations and
docstrings that describe the main ideas without heavy mathematics.
"""

from . import dataset, quantum_kernel, train, plots

__all__ = ["dataset", "quantum_kernel", "train", "plots"]
