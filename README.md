# L4tt1ce

A lightweight **2D** Python implementation of the Lattice Boltzmann Method (LBM).

This project uses the **BGK** (Bhatnagar-Gross-Krook) operator to simulate:

- Navier-Stokes fluid dynamics
- Scalar transport (advection-diffusion)

---

## Requirements
The project uses [poetry](https://python-poetry.org/) for dependency management.
First install poetry as described in the [official documentation](https://python-poetry.org/docs/#installation), then installing the dependencies is as simple as running:

```bash
poetry install
```

This automatically creates a virtual environment and installs all the required packages.
It's recommended to run `poetry config virtualenvs.in-project true` before installing the dependencies, which places the created virtual environment in the project directory (`.venv`).

## Running the examples
There's generally two ways to run examples through the command line, either by activating the virtual environment and running the script directly, or by using poetry to run the script in the virtual environment.

**Running the script using python in the virtual environment**:
```bash
poetry shell
python showcase_heatedObject.py
```

**Running the script through poetry**:
```bash
poetry run python showcase_heatedObject.py
```


## Feature Overview

| Feature                          | Status   | Description                                        |
|----------------------------------|----------|----------------------------------------------------|
| 2D Simulation                    | ✅ Done  | Current implementation supports 2D flows           |
| Dirichlet & Neumann BCs          | ✅ Done  | Velocity, pressure, concentration, flux boundaries |
| Coupled Scalar Transport         | ✅ Done  | Advection-diffusion modeling                       |
| 3D Simulation                    | 🚧 WIP   | Planned for future release                         |
| Geometry from Images             | 🔜 Todo  | Planned feature                                    |
| MRT/TRT Models                   | 🔜 Todo  | Advanced collision operators                       |

## Showcases
### Flow around obstacles
![me](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExdmE5N2tnaDZjN2Jkd3JvaXBxZXMwbTN0YXhtZXB6dTM4bmQwb2UzZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/gv1LI63WFKrTyV4Rcf/giphy.gif)
### Heated obstacle in channel
![me](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExancxc3d4Mmc0YWJxaG5hMGs4eHY3cWZ3ZmxkOTF3dnQ4Z2Z5ZnprYyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/8oFlu2HEWHScOQqiny/giphy.gif)




