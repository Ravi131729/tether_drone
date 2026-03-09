# Tether Drone Simulation (LGVI)

This repository contains a simulation platform for a tethered drone using a **Lie Group Variational Integrator (LGVI)** formulation.

The model includes:
- Multibody tether/string dynamics with discretized segments
- Drone translational and rotational dynamics
- Structure-preserving time stepping via variational integration
- Winch/tether terms in the residual formulation

## Repository Overview

- `main.py`: Entry point to configure parameters and run simulation
- `simulation/`: Core physics and numerical routines
- `simulation/integrator.py`: Time-stepping loop (`run_simulation`)
- `simulation/residual.py`: Translational and attitude residual equations
- `simulation/newton_iter.py`: Newton solver used at each time step
- `plotter/`: Plot and animation utilities
- `models/Assembly.STL`: 3D model used by animation utilities
- `*.npz`: Saved trajectory/result files
- `*.mp4`: Rendered simulation videos

## Quick Start

1. Install Python dependencies (at minimum: `jax`, `numpy`, `matplotlib`).
2. Run:

```bash
python main.py
```

This will run the simulation and save output (for example `free_fall.npz`).

## Output and Visualization

- Numerical trajectories are stored in `.npz` files.
- Plotting/animation helpers are available in `plotter/`.
- Existing demo videos are available as `.mp4` files in the repository root.

## Two Tether-Drone Simulations

Add your final GIFs in the paths below (or update paths to match your files).

### Simulation 1: Free-Fall / Baseline

![Simulation 1 GIF placeholder](./assets/simulation_1.gif)

Description: Baseline tether-drone response under current LGVI setup.

### Simulation 2: Winch/Actuation Scenario

![Simulation 2 GIF placeholder](./assets/simulation_2.gif)

Description: Tether length/winch-influenced behavior and coupled drone motion.

## Notes

- The main script currently runs on CPU (`jax_platforms = "cpu"`).
- You can modify `L`, `N`, `h`, `tf`, thrust, and torque parameters in `main.py`.
