# 📦 **wireflux**

### *Wire-simulation physics engine using Biot–Savart, curvature-driven remeshing, and PyVista visualization.*

---

## 🚀 Overview

**wireflux** is a physics simulation toolkit for evolving current-carrying wires using JxB forces in 3D space.
It implements:

* **Biot–Savart magnetic fields**
* **J × B Lorentz forces**
* **Finite-difference and spline-based curvature evaluation**
* **Adaptive remeshing to maintain filament quality**
* **Mass-conserving wire evolution**
* **PyVista-based 3D visualization**
* A clean, modern package architecture suitable for custom extensions

The toolkit supports both the legacy **Wire** class (finite-difference based) and the modern **NewWire** class (remeshing based), enabling backwards compatibility and forward development.

---

## 📁 Project Structure

```
wireflux/
│
├── src/wireflux/
│   ├── core/        # Engine, State, integrators
│   ├── models/      # Wire, NewWire, filament models
│   ├── physics/     # Biot-Savart, forces, inductance
│   ├── utils/       # geometry, smoothing, grids, vector tools
│   ├── viz/         # PyVista visualization helpers
│   └── __init__.py
│
├── tests/           # pytest suite
└── examples/        # example scripts & notebooks
```

---

# 🔧 Installation

### **1. Clone the repository**

```bash
git clone https://github.com/magnus-haw/wire-simulation.git
cd wireflux
```

---

## **2. Install in development mode (recommended)**

Since wireflux uses a `src/` layout, you should install it in editable mode:

```bash
pip install -e .
```

This allows:

* imports from anywhere (`import wireflux`)
* live code editing
* running examples/tests without path hacks

---

## **3. Optional extras**

### Visualization dependencies (PyVista):

```bash
pip install pyvista pyvistaqt
```

### Developer tooling:

```bash
pip install -e ".[dev]"
```

This includes:

* `pytest`, `pytest-cov`
* `black`, `flake8`, `isort`

---

# 🧪 Running the Test Suite

```bash
pytest -q
```

or with coverage:

```bash
pytest --cov=wireflux
```

The test suite checks:

* Wire vs NewWire force consistency
* curvature correctness
* mass conservation under evolution
* inductance symmetry
* engine time-stepping stability

---

# 🌀 Quick Start Example

### Visualize a circular wire with PyVista:

```python
import numpy as np
import pyvista as pv
from wireflux import Wire

N = 200
t = np.linspace(0, 2*np.pi, N)

path = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
v = np.zeros_like(path)
mass = np.ones((N,1))

w = Wire(path, v, mass, I=1.0, r=0.05)

plotter = pv.Plotter()
w.show(plotter=plotter)
plotter.show()
```

Produces a smoothed tubular visualization of the wire.

---

# ⚡ Using the Engine

A minimal time-stepping example:

```python
import numpy as np
from wireflux import Wire, MultiWireEngine, State

# Define a loop
N = 200
theta = np.linspace(0, 2*np.pi, N)
path = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])

v = np.zeros_like(path)
mass = np.ones((N,1))

wire = Wire(path, v, mass, I=1.0)
state = State("example", [wire], time=0.0, load=0)
engine = MultiWireEngine(state)

# Step 50 times
for _ in range(50):
    engine.step(dt=0.01)
```

---

# 📚 Documentation & Examples

Example scripts live in:

```
examples/
```

Including:

* `visualize_single_wire.py`
* `parallel_wires_force.py`
* `adaptive_remeshing_demo.py`
* `engine_time_evolution.py`
* (optional) notebooks under `examples/notebook/`

These scripts demonstrate the engine, visualization, remeshing, and physics.

---

# 🏗 Development Guide

### Format code:

```bash
black .
isort .
```

### Run lint checks:

```bash
flake8 src/
```

### Run specific tests:

```bash
pytest tests/test_wire_vs_newwire.py
```

### Refresh editable install after reorganizing directories:

```bash
pip install -e .
```

---

# 🧩 Citing wireflux

If you use wireflux in published research, please cite:

```
Haw, M. (2025). wireflux: Adaptive wire-filament simulation engine.
https://github.com/yourusername/wireflux
```

---

# 🤝 Contributing

Pull requests are welcome!

If contributing:

1. fork the repo
2. create a feature branch
3. write tests
4. run `pytest`
5. submit PR

---

# 📄 License

MIT License — feel free to use in scientific, academic, or commercial projects.

---

# 🔮 Roadmap

* GPU-accelerated Biot–Savart (Numba/CUDA)
* Implicit curvature-flow integrators
* Wire-wire interaction via fast multipole methods
* Improved PyVista visualizer with curvature coloring
* Interactive Jupyter notebook examples

---

If you'd like, I can also generate:

* A Sphinx documentation tree
* A mkdocs site
* A full example gallery (like scikit-image or PyVista)
* A logo or badge set for wireflux
* A CONTRIBUTING.md file

Just tell me!
