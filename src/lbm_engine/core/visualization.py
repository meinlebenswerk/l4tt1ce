# Visualization functions, not yet generalized

import matplotlib.pyplot as plt
import os 
from pyevtk.hl import imageToVTK
import numpy as np
from numpy.typing import NDArray

""" This file contains some basic visualization files that can create plots using matplotlib as well as VTK Files for Paraview
The visualization methods are kinda ugly and have to be rewritten because of scaling and layout but it works for now so..... """

def visualize(sim, t: int, vmax_set):
    vmag = np.sqrt(sim.u[:, :, 0]**2 + sim.u[:, :, 1]**2)
    if "obstacle" in sim.geometry:
        mask = sim.geometry["obstacle"].mask
        mask = np.transpose(mask) if mask.shape != vmag.shape else mask
        vmag[mask] = np.nan
    # plt.figure(figsize=(8, 5))
    u_mag = np.sqrt(sim.u[:, :, 0]**2 + sim.u[:, :, 1]**2)
    max_velocity = np.max(u_mag)
    # print(max_velocity)
    plt.imshow(vmag, cmap="coolwarm", vmin=0, vmax=vmax_set)
    plt.title(f"Velocity magnitude – t = {t}")
    plt.axis("off")
    plt.colorbar(label="|u|")
    plt.tight_layout()
    plt.savefig(f"frames/frame_{t:05d}.png", dpi=150)
    plt.close()

def visualizeScalar(sim, t: int):
    u_mag = np.sqrt(sim.phi[:, :]**2)
    max_velocity = np.max(u_mag)
    print(max_velocity)
    plt.figure(figsize=(8, 4))
    plt.imshow(sim.phi, cmap='inferno')
    plt.title(f"Scalar field ϕ at t={t}")
    plt.colorbar(label="ϕ")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"frames/phi_{t:05d}.png", dpi=150)
    plt.close()

def visualize_combined(sim, t: int, overall_title: str | None = None):
    # this creates matplotlib plots for velocity and scalar value of a single lattice setup
    # maybe this has to be changed to receive fields and not a lattice object for more customizability
    vmag = np.sqrt(sim.u[:, :, 0]**2 + sim.u[:, :, 1]**2)
    phi = sim.phi

    # We mask the obstabcles as NaN, so they are more visible in the plot area
    if "obstacle" in sim.operators:
        mask = sim.operators["obstacle"].mask
        if mask.shape != vmag.shape:
            mask = mask.T
        vmag[mask] = np.nan
        phi[mask] = np.nan

    # basic plot shit
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # We set a overall title for the plot
    if overall_title:
        fig.suptitle(overall_title, fontsize=14, fontweight='bold', y=0.98)

    # This is plots the magnitude velocity and sets upper limit as some fixed value from parameter (can be changed somehow but VTK is better anyways)
    im1 = axs[0].imshow(vmag, cmap="coolwarm", vmin=0, vmax=0.19)
    axs[0].set_title("Velocity Magnitude")
    axs[0].axis("off")
    fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04, label="|u|")

    # This plots the scalar value and autoscales (for dirichlet boundaries it will result in a constant scaling because of the maximum being a fixed value)
    im2 = axs[1].imshow(phi, cmap="inferno")
    axs[1].set_title("Scalar Field ϕ")
    axs[1].axis("off")
    fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04, label="ϕ")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("frames", exist_ok=True)
    plt.savefig(f"frames/combined_{t:05d}.png", dpi=150)
    plt.close()

def export_fields_vti(timestep: int, u: NDArray, phi: NDArray | None = None, output_dir: str = "output_vti"):
    #this creates vti files for paraview for velocity and (scalar field)
    ny, nx = u.shape[:2]
    spacing = (1.0, 1.0, 1.0) 
    origin = (0.0, 0.0, 0.0)

    # Prepare fields as (nx, ny, 1)
    # some disgusting shaping going on here, dont ask me how this works or why (it was painful)
    u_x = np.ascontiguousarray(u[:, :, 0].T[:, :, np.newaxis])
    u_y = np.ascontiguousarray(u[:, :, 1].T[:, :, np.newaxis])
    u_z = np.zeros_like(u_x)

    point_data = {
        "velocity": (u_x, u_y, u_z)
    }
    if phi is not None:
        phi_3d = np.ascontiguousarray(phi.T[:, :, np.newaxis])
        point_data["phi"] = phi_3d

    os.makedirs(output_dir, exist_ok=True)

    imageToVTK(
        os.path.join(output_dir, f"fields_{timestep:05d}"),
        origin=origin,
        spacing=spacing,
        pointData=point_data
    )



