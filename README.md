<div align="center">

# SchrödArt 🌌

**An interactive web app that solves the Schrödinger equation numerically and turns the results into generative art.**

[Live demo](https://gnm4pxwnrpb6cy3syst6sn.streamlit.app) · [Technical reports](https://github.com/Alyaa203/Liverable)

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

## Overview

SchrödArt simulates how a quantum particle behaves (the Schrödinger equation) and shows the results as interactive scientific plots and as artwork.

**Why it exists:** it is an individual engineering project at ENSC (Bordeaux INP). The goal was to implement two independent numerical methods, check them against each other, and make the physics visual and accessible through a web interface.

**At a glance:**
- Two numerical solvers built from scratch: **modal decomposition** (sparse matrix diagonalisation) and **Split-Step Fourier** (FFT)
- **Cross-validation** between the two methods, with a relative error of about 2–4%
- Energy spectra turned into **5 styles of generative art**
- Deployed online with Streamlit Cloud

$$i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2\psi + V(x,t)\psi$$

---

## Features

| Tab | What it does |
| --- | --- |
| **2D stationary regime** | Builds the 2D Hamiltonian, computes the lowest-energy eigenstates and shows each mode and the Gaussian potential |
| **1D time-dependent regime** | Animates how the probability density evolves over time, with a 3D surface view |
| **Quantum art** | Generates images from the computed eigenvalues: rosette, nebula, crystal, fractal mandala, spiral galaxy |
| **Split-Step Fourier** | Simulates a wave packet step by step in 3 scenarios: **tunnelling** through a barrier, **harmonic oscillator**, **double well** |
| **Cross-validation** | Runs both methods on the same problem and reports the relative error between them |

All physical parameters (grid size, number of modes, potential position, width and depth, time) can be changed live with sliders.

---

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python 3.11 |
| Numerical computing | NumPy (vectors, FFT), SciPy (sparse matrices, `eigsh` / ARPACK, `eigh_tridiagonal`) |
| Visualisation | Matplotlib, Pillow (GIF export) |
| Web interface | Streamlit |
| Deployment | Streamlit Community Cloud, Dev Container (GitHub Codespaces) |

### Numerical methods in brief

- **Modal decomposition** (`simulation.py`): the 2D Hamiltonian is built with a Kronecker product of finite-difference matrices, $H = -\tfrac{1}{2}(D_x \otimes I + I \otimes D_y) + V$, then diagonalised with sparse ARPACK. The time evolution is exact for each mode, $\psi(x,t) = \sum_j c_j \psi_j(x) e^{-iE_j t}$, so the norm is conserved.
- **Split-Step Fourier** (`Fourier.py`): second-order Strang splitting, alternating half-steps in position space and full steps in momentum space via FFT.

---

## Getting started

**Easiest:** open the [live demo](https://gnm4pxwnrpb6cy3syst6sn.streamlit.app). It may take a minute to wake up.

**Run locally** (Python 3.9+):

```bash
git clone https://github.com/Alyaa203/P2i.git
cd P2i
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open http://localhost:8501.

> **Note:** open the *2D stationary* or *1D time-dependent* tab first. The *Quantum art* tab uses the eigenvalues they compute.

You can also open the repo in **GitHub Codespaces**: the Dev Container installs everything and starts the app automatically.

### Project structure

```
P2i/
├── streamlit_app.py   # Web interface (5 tabs)
├── simulation.py      # Modal method (2D stationary + 1D time-dependent)
├── Fourier.py         # Split-Step Fourier method
├── visualisation.py   # Generative art (5 styles)
└── requirements.txt
```

---

## References

- D. J. Griffiths, *Introduction to Quantum Mechanics*, 3rd ed., Cambridge University Press, 2018.
- D. J. Tannor, *Introduction to Quantum Mechanics: A Time-Dependent Perspective*, University Science Books, 2007.
- G. Strang, "On the construction and comparison of difference schemes", *SIAM J. Numer. Anal.*, 1968.

---

**Author:** Alyaa Saab, engineering student at ENSC (Bordeaux INP)
