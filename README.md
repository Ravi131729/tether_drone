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





## Notes

- The main script currently runs on CPU (`jax_platforms = "cpu"`).
- You can modify `L`, `N`, `h`, `tf`, thrust, and torque parameters in `main.py`.
