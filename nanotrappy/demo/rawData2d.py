# -*- coding: utf-8 -*-
import nanotrappy as nt
import os
import numpy as np
import matplotlib.pyplot as plt
from nanotrappy.utils.physicalunits import *

if __name__ == "__main__":
    # Folder for data storage
    datafolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testfolder")

    # Define beams
    red_beam = nt.BeamPair(937e-9, 0.3 * mW, 937e-9, 0.4 * mW)
    blue_beam = nt.BeamPair(685.5e-9, 5 * mW, 685.6e-9, 4.5 * mW)

    # Define the trap
    trap = nt.Trap_beams(blue_beam, red_beam)

    # Atomic system and surface
    syst = nt.atomicsystem(nt.Caesium(), "6S1/2", f3)
    surface = nt.CylindricalSurface(axis=nt.AxisZ(), radius=220e-9)

    # Initialize simulation
    Simul = nt.Simulation(syst, nt.SiO2(), trap, datafolder)
    Simul.geometry = nt.PlaneXY(normal_coord=0)
    Simul.compute()

    # Access the raw data (potentials attribute or Etot depending on need)
    raw_2d_data = Simul.total_potential()

    # Save raw data as .npy file
    output_file = "raw_2d_trap_data.npy"
    np.save(output_file, raw_2d_data)
    print(f"2D raw trap data saved to: {output_file}")

    # Check data dimensions and preview
    print("Shape of raw data:", raw_2d_data.shape)
    print("Sample data (top-left corner):")
    print(raw_2d_data[:5, :5])

    # Optional: Visualize as heatmap
    selected_mf_state = 0  # Index of the mf state to display. Note that it's not the same as mf since index can only be positive.
    data_2d = raw_2d_data[:, :, selected_mf_state]
    print(np.shape(data_2d))
    min_value = np.min(data_2d)  # Find the minimum value in the 2D array
    min_position = np.unravel_index(np.argmin(data_2d), data_2d.shape)  # Get coordinates of min value
    data_2d_slice = data_2d[100,:]
    val2 = np.min(data_2d_slice)


    print(f"  Lowest potential value: {min_value}")
    print(f"  Position (row, col): {min_position}")
    print(f"  Val2: {val2}")
    plt.imshow(data_2d, extent=[-800, 800, -800, 800], origin='lower', aspect='auto', cmap="viridis")
    plt.colorbar(label="Potential")
    plt.title("2D Raw Trap Data")
    plt.xlabel("X axis (nm)")
    plt.ylabel("Y axis (nm)")
    plt.show()
