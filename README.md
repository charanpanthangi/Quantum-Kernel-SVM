# Quantum Kernel SVM — Simple Beginner-Friendly Demo

This project shows how quantum feature maps can help separate tricky shapes
like moons or circles. The code is designed for beginners with clear comments
and a simple layout.

## 1. What this project does
- Generates a toy dataset shaped like two moons (or circles).
- Encodes the data into a quantum state using a Qiskit feature map.
- Computes a **quantum kernel matrix** that captures non-linear patterns.
- Trains two Support Vector Machines (SVMs): one classical (RBF) and one using
the quantum kernel.
- Compares their accuracies and saves helpful plots.

## 2. Why quantum helps with non-linear data
Quantum feature maps place data into a very high-dimensional space using
superposition and entanglement. In that space, shapes that look tangled in two
dimensions become easier to separate with a simple boundary.

## 3. Step-by-step intuition (simple English)
- We create a small dataset shaped like moons or circles.
- The quantum feature map turns each 2D point into a pattern of qubit angles.
- Qubits can be in many states at once (superposition) and linked together
  (entanglement), so the data spreads out into a rich space.
- A kernel matrix measures how similar every pair of points is after that
  quantum encoding.
- An SVM uses the kernel to draw a curved boundary that wraps around the moons.

## 4. Repository structure
```
quantum-kernel-svm-demo/
├── app/
│   ├── __init__.py
│   ├── quantum_kernel.py      # Quantum feature map + kernel computation
│   ├── dataset.py             # Dataset generation helpers
│   ├── train.py               # Training classical and quantum SVMs
│   ├── plots.py               # Plot dataset and decision boundaries
│   ├── main.py                # CLI entry point
├── notebooks/
│   └── quantum_kernel_demo.ipynb  # Optional Jupyter notebook walkthrough
├── examples/
│   ├── sample_moons.svg       # Scatter plot of the dataset (text-based SVG)
│   └── sample_results.svg     # Side-by-side decision boundaries (text-based SVG)
├── tests/
│   ├── test_quantum_kernel.py
│   └── test_training.py
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
└── LICENSE
```

## 5. How to install
1. Make sure Python 3.11+ is available.
2. Create and activate a virtual environment (optional but recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 6. How to run CLI
Run the full pipeline from the repository root:
```bash
python -m app.main --dataset moons
```
This will generate data, train both models, and save plots inside `examples/`.

## 7. How to run notebook
Launch Jupyter and open the provided notebook:
```bash
jupyter notebook notebooks/quantum_kernel_demo.ipynb
```
The notebook walks through the same steps with extra explanations.

## 8. Results comparison (Q-SVM vs classical RBF)
- The classical RBF SVM already handles curves well.
- The quantum kernel SVM can capture extra twists by using quantum state
  overlaps. On small toy data, the accuracies are usually close, but the plot
  shows how the boundary can flex differently.

## 9. Sample images
- `examples/sample_moons.svg` shows the dataset.
- `examples/sample_results.svg` shows both decision boundaries.

## 10. Limitations & future improvements
- Simulations can be slow for large datasets because quantum kernels scale with
  the number of samples.
- The demo uses a simulator; running on real hardware would introduce noise.
- Future ideas: try deeper feature maps, add more datasets, or explore kernel
  alignment techniques.

Enjoy exploring quantum kernels!
