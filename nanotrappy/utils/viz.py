from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from operator import itemgetter
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.signal import find_peaks
from nanotrappy.utils.physicalunits import *
from nanotrappy.utils.utils import *
from nanotrappy.utils.lineslicer import LineSlice
import re
import itertools
import mplcursors
import time


_custom_highlight_kwargs = dict(
    # Only the kwargs corresponding to properties of the artist will be passed.
    # Line2D.
    color="tab:red",
    markeredgecolor="tab:red",
    linewidth=3,
    markeredgewidth=3,
    # PathCollection.
    facecolor="tab:red",
    edgecolor="tab:red",
)

_custom_annotation_kwargs = dict(
    bbox=dict(
        boxstyle="round,pad=.5",
        fc="#F6D3D4",  # "tab:red",
        alpha=1,
        ec="#F6D3D4",  # "k",
    ),
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3",
        shrinkB=0,
        ec="#F6D3D4",  # "k",
        fc="#F6D3D4",
    ),
)

# font = {"size": 22}
import matplotlib

# matplotlib.rc("font", **font)
matplotlib.rcParams["axes.unicode_minus"] = False

from PyQt5.QtCore import QFile, QThread, pyqtSignal, QObject, pyqtSlot


class Viz(QObject):
    """Class that contains all the visualization methods.

    Attributes:
        simul (Simulation object): Simulation object contaning a trap,
            a system and a surface. For most of the methods in the class,
            the simulations have to be run beforehand.
        trapping_axis (str): Axis perpendicular to the structure along which
            we want to trap the atoms. Important for the 3 1D plot method.
            Either "X", "Y" or "Z".

    """

    _signal = pyqtSignal()
    benchmark = pyqtSignal(float)
    finished = pyqtSignal()

    def __init__(self, simul, trapping_axis):
        # Trapping_axis is the one perpendicular to the surface if one is defined
        super().__init__()
        self.trapping_axis = trapping_axis
        self.simul = simul

    def plot_trap(self, plane, mf=0, Pranges=[10, 10], increments=[0.1, 0.1]):
        """Shows a 2D plot of the total potential with power sliders
        Only available if a 2D simulation has been run.

        Args:
            plane (str): As we are dealing with 2D plots, we have to specify
            the plane we are looking at to choose the right coordinates for plotting.
            mf (int): Mixed mf state we want to plot. In 2D we can only
            specify one integer. Default to 0.
            Pranges (list): List with the maximum values of the beam powers
            we want to display on the sliders. Defaults to [10,10]
        Raise:
            TypeError: if only a 1D computation of the potential has been
            done before plotting.

        Returns:
            (tuple): containing:

                - fig: figure
                - ax: axis of figure
                - slider_ax: sliders (needed for interactivity of the sliders)

        """
        if np.ndim(self.simul.total_potential()) <= 2:
            raise TypeError("This method can only be used if a 2D computation of the potential has been done")

        if len(Pranges) != len(self.simul.trap.beams):
            raise ValueError(
                "When specifying the upper ranges of P for plotting, you have to give as many as many values as there are beams."
            )

        _, mf = check_mf(self.simul.atomicsystem.f, mf)
        coord1, coord2 = set_axis_from_plane(plane, self.simul)
        mf_index = int(mf + self.simul.atomicsystem.f)
        trap = np.real(self.simul.total_potential())[:, :, mf_index]
        trap_noCP = np.real(self.simul.total_potential_noCP[:, :, mf_index])
        fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1)
        plt.subplots_adjust(left=0.5, bottom=0.1)
        # the norm TwoSlopeNorm allows to fix the 0 of potential to the white color, so that we can easily distinguish between positive and negative values of the potential
        a = ax.pcolormesh(
            coord1 / nm,
            coord2 / nm,
            np.transpose(trap),
            shading="gouraud",
            norm=colors.TwoSlopeNorm(
                vmin=min(np.min(trap_noCP), -0.001), vcenter=0, vmax=max(np.max(trap_noCP) * 2, 0.001)
            ),
            cmap="seismic_r",
        )
        cbar = plt.colorbar(a)
        cbar.set_label("Total potential (mK)", rotation=270, labelpad=12, fontsize=14)

        ax.set_xlabel("%s (nm)" % (plane[0].lower()), fontsize=14)
        ax.set_ylabel("%s (nm)" % (plane[1].lower()), fontsize=14)
        plt.setp(ax.spines.values(), linewidth=1.5)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.set_title("2D plot of trapping potential \n for mf = %s in the %s plane" % (mf, plane), fontsize=18)

        ax.margins(x=0)
        axcolor = "lightgoldenrodyellow"
        slider_ax = []
        axes = []

        for (k, beam) in enumerate(self.simul.trap.beams):
            axes.append(plt.axes([0.15 + k * 0.08, 0.1, 0.03, 0.75], facecolor=axcolor))
            slider_ax.append(
                Slider(
                    axes[k],
                    "Power \n Beam %s (mW)" % (k + 1),
                    0,
                    Pranges[k],
                    valinit=self.simul.trap.beams[k].get_power() * 1e3,
                    valstep=increments[k],
                    orientation="vertical",
                )
            )

        def updateP(val):
            P = []
            for (k, slider) in enumerate(slider_ax):
                P.append(slider.val * mW)
            self.simul.trap.set_powers(P)
            trap_2D = self.simul.total_potential()[:, :, mf_index]
            a.set_array(np.transpose(np.real(self.simul.total_potential_noCP[:, :, mf_index])).ravel())
            a.autoscale()
            a.set_array(np.transpose(np.real(trap_2D)).ravel())
            fig.canvas.draw_idle()

        for slider in slider_ax:
            slider.on_changed(updateP)

        s1, s2 = np.transpose(trap).shape
        LnTr = LineSlice(a, s1, s2, coord1 / nm, coord2 / nm, ax2)

        plt.show()

        return fig, ax, slider_ax

    def plot_trap1D(self, axis, mf=0, Pranges=[10, 10], increments=[0.1, 0.1]):
        """Shows a 1D plot of the total potential with power sliders
        Only available if a 1D simulation has been run.

        Args:
            axis (str): As we are dealing with 1D plots, we have to specify
                the axis along which we are looking the trap.
            mf (int or list): Mixed mf state we want to plot. If a list is given,
                all the specified mf states will be showed. Default to 0.
            Pranges (list): List with the maximum values of the beam powers we
                want to display on the sliders. Defaults to [10,10]
        Raise:
            TypeError: if only a 2D computation of the potential has been done
            before plotting.

        Returns:
            (tuple): containing:

                - fig: figure
                - ax: axis of figure
                - slider_ax: sliders (needed for interactivity of the sliders)
        """

        if np.ndim(self.simul.total_potential()) >= 3:
            raise TypeError("This method can only be used if a 1D computation of the potential has been done.")
        if len(Pranges) != len(self.simul.trap.beams):
            raise ValueError(
                "When specifying the upper ranges of P for plotting, you have to give as many as many values as there are beams."
            )

        _, mf = check_mf(self.simul.atomicsystem.f, mf)

        x = set_axis_from_axis(axis, self.simul)
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.27)
        jet = cm = plt.get_cmap("Greys")
        cNorm = colors.Normalize(vmin=-1, vmax=len(mf))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        a = []

        mf_index = mf + [self.simul.atomicsystem.f]

        trap = np.real(self.simul.total_potential())
        trap_noCP = np.real(self.simul.total_potential_noCP)
        ax.set_xlabel("%s (nm)" % (axis.lower()), fontsize=14)
        ax.set_ylabel("E (mK)", fontsize=14)
        plt.setp(ax.spines.values(), linewidth=1.5)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=2)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.set_title("1D plot of trapping potential \n for mf = %s along %s " % (mf, axis), fontsize=18)

        for k in range(len(mf_index)):
            colorVal = "k"  # scalarMap.to_rgba(k)
            a = a + plt.plot(
                x / nm,
                trap[:, mf_index[k]],
                color=colorVal,
                label="m$_f$ = %s" % (mf[k]),
                linewidth=2 + 3 / len(self.simul.mf_all),
            )

        if len(mf) == 1 and len(self.simul.trap.beams) == 2:
            (b,) = plt.plot(
                x / nm,
                np.real(self.simul.trap.beams[0].get_power() * np.real(self.simul.potentials[0, :, mf_index[0]])),
                color="blue",
                linewidth=2,
            )
            (r,) = plt.plot(
                x / nm,
                np.real(self.simul.trap.beams[1].get_power() * np.real(self.simul.potentials[1, :, mf_index[0]])),
                color="red",
                linewidth=2,
            )
        else:
            pass
            # plt.legend()

        axcolor = "lightgoldenrodyellow"
        slider_ax = []
        axes = []
        for (k, beam) in enumerate(self.simul.trap.beams):
            axes.append(plt.axes([0.25, 0.15 - k * 0.1, 0.6, 0.03], facecolor=axcolor))
            slider_ax.append(
                Slider(
                    axes[k],
                    "Power \n Beam %s (mW)" % (k + 1),
                    0,
                    Pranges[k],
                    valinit=self.simul.trap.beams[k].get_power() * 1e3,
                    valstep=increments[k],
                )
            )
            slider_ax[k].label.set_size(14)

        cursor = mplcursors.cursor(
            a, highlight=True, highlight_kwargs=_custom_highlight_kwargs, annotation_kwargs=_custom_annotation_kwargs
        )

        @cursor.connect("add")
        def on_add(sel):
            artist = sel.artist
            label = artist.get_label() or ""
            mf = self.simul.atomicsystem.f + int(label.split()[2])

            label = f"Choice : {label}"
            idx = int(sel.target.index)

            temp_vec = self.simul.total_vecs[idx, mf]
            temp_vec = np.abs(temp_vec) ** 2
            decomp = f"State : {vec_to_string(temp_vec)}"

            x, y = sel.target
            textx = f"x = {x:.1f} nm"
            texty = f"y = {y:.2f} mK"

            size = max(len(textx), len(texty), len(decomp))
            label = label.center(size, "-")
            text = f"{label}\n{textx}\n{texty}\n{decomp}"
            sel.annotation.set_text(text)

        def updateP(val):
            for selection in cursor.selections:
                cursor.remove_selection(selection)
            P = []
            for (k, slider) in enumerate(slider_ax):
                P.append(slider.val * mW)
            self.simul.trap.set_powers(P)
            trap = np.real(self.simul.total_potential())
            for k in range(len(mf)):
                trap_k = trap[:, mf_index[k]]
                a[k].set_ydata(trap_k)

            if len(mf) == 1 and len(self.simul.trap.beams) == 2:
                b.set_ydata(
                    np.real(self.simul.trap.beams[0].get_power() * np.real(self.simul.potentials[0, :, mf_index[0]]))
                )
                r.set_ydata(
                    np.real(self.simul.trap.beams[1].get_power() * np.real(self.simul.potentials[1, :, mf_index[0]]))
                )
            fig.canvas.draw_idle()

        for slider in slider_ax:
            slider.on_changed(updateP)
        plt.show()

        return fig, ax, slider_ax

    def plot_3axis(self, coord1, coord2, mf=0, Pranges=[10, 10], increments=[0.1, 0.1]):
        """Shows 3 1D plots of the total potential with power sliders,
        and trapping frequencies for each axis if possible.
        Starts by simulating a 1D trap along the trapping_axis attribute of
        the Viz object and finds the minimum.
        Once found, simulates 1D traps in the 2 other orthogonal directions
        and finds the associated frequency.
        When looking at nanostructure with possible different trapping axis
        (like nanofibers), a new Viz object has to be defined
        in order to use this method.

        Args:
            coord1 (float): First coordinate on the orthogonal plane to the
                trapping axis. If trapping axis is Y, coord1 should be the one on X.
            coord2 (float): Second coordinate on the orthogonal plane to the
                trapping axis.
            mf (int): integer between -F and +F. No list possible here.
            Pranges (list): List with the maximum values of the beam powers we
                want to display on the sliders. Defaults to [10,10]

        Returns:
            (tuple): containing:

                - fig: figure
                - ax: axis of figure
                - slider_ax: sliders (needed for interactivity of the sliders)
        """
        _, mf = check_mf(self.simul.atomicsystem.f, mf)

        if len(mf) > 1:
            raise ValueError("This 3D plot can only be done for one specific mf at a time")

        mf_shift = mf + self.simul.atomicsystem.f
        axis_of_interest, axis1, axis2 = get_sorted_axis(self.trapping_axis, self.simul)
        axis1_name, axis2_name = get_sorted_axis_name(self.trapping_axis)
        axis_index, axis1_index, axis2_index = set_axis_index_from_axis(self.trapping_axis)
        axis_name_list = [self.trapping_axis, axis1_name, axis2_name]

        if len(self.simul.E[0].shape) != 4:
            print("[WARNING] 3D Electric fields must be fed in the Simulation class in order to use this function")
        else:
            mf_index, edge, y_outside, trap_1D_Y_outside = self.get_coord_trap_outside_structure(
                self.trapping_axis, coord1, coord2, mf, edge_no_surface=None
            )
            ymin_ind, y_min, trap_depth, trap_prominence = self.get_min_trap(y_outside, trap_1D_Y_outside)
            omegax, omegay, omegaz = 0, 0, 0
            if not np.isnan(y_min):
                min_pos = np.zeros(3)
                min_pos[axis_index] = y_min + edge
                min_pos[axis1_index] = coord1
                min_pos[axis2_index] = coord2
                omegay = self.get_trapfreq(y_outside, trap_1D_Y_outside)
                _, _, x_outside, trap_1D_X_allw = self.get_coord_trap_outside_structure(
                    axis1_name,
                    np.delete(min_pos, axis1_index)[0],
                    np.delete(min_pos, axis1_index)[1],
                    mf,
                    edge_no_surface=None,
                )
                omegax = self.get_trapfreq(x_outside, trap_1D_X_allw)
                _, _, z_outside, trap_1D_Z_allw = self.get_coord_trap_outside_structure(
                    axis2_name,
                    np.delete(min_pos, axis2_index)[0],
                    np.delete(min_pos, axis2_index)[1],
                    mf,
                    edge_no_surface=None,
                )
                omegaz = self.get_trapfreq(z_outside, trap_1D_Z_allw)

            fig, ax = plt.subplots(3, figsize=(15, 10))
            plt.subplots_adjust(left=0.25)
            axcolor = "lightgoldenrodyellow"
            props = dict(boxstyle="round", facecolor=axcolor, alpha=0.5)

            textstr = "\n".join(
                (
                    r"$\mathrm{trap \, position}=%.2f (nm) $" % (y_min * 1e9,),
                    r"$\mathrm{depth}=%.2f (mK) $" % (trap_depth,),
                    r"$\omega_%s =%.2f (kHz) $"
                    % (
                        self.trapping_axis,
                        omegay * 1e-3,
                    ),
                    r"$\omega_%s =%.2f (kHz) $"
                    % (
                        axis1_name,
                        omegax * 1e-3,
                    ),
                    r"$\omega_%s =%.2f (kHz) $"
                    % (
                        axis2_name,
                        omegaz * 1e-3,
                    ),
                )
            )

            box = plt.text(
                -0.3, 0.6, textstr, transform=ax[2].transAxes, fontsize=14, verticalalignment="top", bbox=props
            )

            slider_ax = []
            axes = []
            for (k, beam) in enumerate(self.simul.trap.beams):
                axes.append(plt.axes([0.1 + k * 0.05, 0.32, 0.03, 0.5], facecolor=axcolor))
                print(self.simul.trap.beams[k].get_power())
                slider_ax.append(
                    Slider(
                        axes[k],
                        "Power \n Beam %s \n (mW)" % (k + 1),
                        0,
                        Pranges[k],
                        valinit=self.simul.trap.beams[k].get_power() * 1e3,
                        valstep=increments[k],
                        orientation="vertical",
                    )
                )

            index_1 = np.argmin(np.abs(axis1 - coord1))
            index_2 = np.argmin(np.abs(axis2 - coord2))

            (ly,) = ax[0].plot(y_outside, trap_1D_Y_outside, linewidth=3, color="darkblue")
            (point,) = ax[0].plot(y_outside[ymin_ind], trap_1D_Y_outside[ymin_ind], "ro")
            ax[0].set_ylim([-2, 2])
            if not np.isnan(y_min):
                (lx,) = ax[1].plot(axis1, trap_1D_X_allw, linewidth=2, color="royalblue")
                (lz,) = ax[2].plot(axis2, trap_1D_Z_allw, linewidth=2, color="royalblue")
                (point1,) = ax[1].plot(axis1[index_1], trap_1D_X_allw[index_1], "ro")
                (point2,) = ax[2].plot(axis2[index_2], trap_1D_Z_allw[index_2], "ro")

            else:
                (lx,) = ax[1].plot(axis1, np.zeros((len(axis1),)), linewidth=2, color="royalblue")
                (lz,) = ax[2].plot(axis2, np.zeros((len(axis2),)), linewidth=2, color="royalblue")

            plt.grid(alpha=0.5)
            for k in range(len(ax)):
                ax[k].set_xlabel("%s (m)" % (axis_name_list[k].lower()), fontsize=14)
                plt.setp(ax[k].spines.values(), linewidth=2)
                ax[k].axhline(y=0, color="black", linestyle="--", linewidth=2)
                ax[k].tick_params(axis="both", which="major", labelsize=14)
            ax[0].set_title("Total dipole trap for mf = %s in the 3 directions" % (mf[0]), fontsize=18)

            fig.text(0.21, 0.5, "Potential (mK)", ha="center", va="center", rotation="vertical", fontsize=14)

            def updateP(val):
                P = []
                for (k, slider) in enumerate(slider_ax):
                    P.append(slider.val * mW)
                self.simul.trap.set_powers(P)
                mf_index, edge, y_outside, trap_1D_Y = self.get_coord_trap_outside_structure(
                    self.trapping_axis, coord1, coord2, mf, edge_no_surface=None
                )
                ymin_ind, y_min, trap_depth, trap_prominence = self.get_min_trap(y_outside, trap_1D_Y)
                print("y_min = ", y_min)
                ax[0].set_ylim([-2, trap_1D_Y.max()])
                axcolor = "lightgoldenrodyellow"
                props = dict(boxstyle="round", facecolor=axcolor, alpha=0.5)
                if not np.isnan(y_min):
                    min_pos = np.zeros(3)
                    min_pos[axis_index] = y_min + edge
                    min_pos[axis1_index] = coord1
                    min_pos[axis2_index] = coord2
                    omegay = self.get_trapfreq(y_outside, trap_1D_Y)
                    _, _, x_outside, trap_1D_X = self.get_coord_trap_outside_structure(
                        axis1_name,
                        np.delete(min_pos, axis1_index)[0],
                        np.delete(min_pos, axis1_index)[1],
                        mf,
                        edge_no_surface=None,
                    )
                    omegax = self.get_trapfreq(x_outside, trap_1D_X)
                    _, _, z_outside, trap_1D_Z = self.get_coord_trap_outside_structure(
                        axis2_name,
                        np.delete(min_pos, axis2_index)[0],
                        np.delete(min_pos, axis2_index)[1],
                        mf,
                        edge_no_surface=None,
                    )

                    omegaz = self.get_trapfreq(z_outside, trap_1D_Z)

                    lx.set_ydata(trap_1D_X)
                    lz.set_ydata(trap_1D_Z)
                    point.set_data(y_outside[ymin_ind], trap_1D_Y[ymin_ind])
                    point1.set_data(axis1[index_1], trap_1D_X[index_1])
                    point2.set_data(axis2[index_2], trap_1D_Z[index_2])

                    ax[1].set_ylim([trap_1D_X.min(), trap_1D_X.max()])
                    ax[2].set_ylim([trap_1D_Z.min(), trap_1D_Z.max()])
                    ax[0].set_ylim([2 * trap_depth, 2 * trap_1D_Y.max()])

                    textstr = "\n".join(
                        (
                            r"$\mathrm{trap \, position}=%.2f (nm) $" % (y_min * 1e9,),
                            r"$\mathrm{depth}=%.2f (mK) $" % (trap_depth,),
                            r"$\omega_%s =%.2f (kHz) $"
                            % (
                                self.trapping_axis,
                                omegay * 1e-3,
                            ),
                            r"$\omega_%s =%.2f (kHz) $"
                            % (
                                axis1_name,
                                omegax * 1e-3,
                            ),
                            r"$\omega_%s =%.2f (kHz) $"
                            % (
                                axis2_name,
                                omegaz * 1e-3,
                            ),
                        )
                    )

                    box.set_text(textstr)
                else:
                    textstr = r"$\mathrm{depth}=%.2f (mK) $" % (trap_depth,)

                    box.set_text(textstr)
                ly.set_ydata(np.squeeze(np.real(trap_1D_Y)))

            for slider in slider_ax:
                slider.on_changed(updateP)
            plt.show()
            return fig, ax, slider_ax

    def get_min_trap(self, y_outside, trap_outside, edge_no_surface=None):
        """Finds the minimum of the trap (ie total_potential()) computed in the simulation object

        Args:
            y_outside (array): truncated coordinates
            trap_outside (array): truncated 1D trap
            axis (str): axis along which we are looking at the trap.
            mf (int or list): Mixed mf state we want to analyze. Default to 0.
            edge_no_surface (float): Position of the edge of the structure.
                Only needed when no Surface is specified.
                When a Surface object is given, it is found automatically with the CP masks.
                Defaults to None.

        Raise:
            TypeError: if only a 2D computation of the potential has been done before plotting.

        Returns:
            (tuple): containing:

                - int: Index of the position of the minimum
                (from the outside coordinate, putting the surface at 0).
                - float: Position of the trap minimum when putting the surface at 0.
                - float: Trap depth (ie, value of the trap at the minimum).
                - float: Height of the potential barrier for the atoms
                (ie difference between the trap depth and the closest local maxima).
                - float: Idx of left prominence if exists
        """

        if np.ndim(trap_outside) >= 3:
            raise TypeError("This method can only be used if a 1D computation of the potential has been done.")

        local_minima = find_peaks(-trap_outside, distance=10, prominence=5e-4)
        if len(local_minima[0]) == 0:
            print("[WARNING] No local minimum found")
            return np.nan, np.nan, 0, 0, np.nan
        elif len(local_minima[0]) == 1 and local_minima[0][0] > 5:
            print("[INFO] One local miminum found at %s" % (y_outside[local_minima[0][0]]))
            return (
                local_minima[0][0],
                y_outside[local_minima[0][0]],
                trap_outside[local_minima[0][0]],
                -local_minima[1]["prominences"][0],
                local_minima[1]["left_bases"][0],
            )
        elif len(local_minima[0]) == 1 and local_minima[0][0] <= 5:
            print("[WARNING] One local minimum found but too close to the edge of the structure")
            return np.nan, np.nan, 0, 0, np.nan
        else:
            arg = np.argmin(np.real(trap_outside[local_minima[0]]))
            print(
                "[WARNING] Many local minima found, taking only the lowest one into account at %s"
                % (y_outside[local_minima[0][arg]])
            )
            return (
                local_minima[0][arg],
                y_outside[local_minima[0][arg]],
                trap_outside[local_minima[0][arg]],
                -local_minima[1]["prominences"][arg],
                local_minima[1]["left_bases"][0],
            )

    def get_trapfreq(self, y_outside, trap_outside, edge_no_surface=None):
        """Finds the value of the trapping frequency (in Hz) along the specified axis

        Args:
            simul (Simulation object): A simulation object with computation of 1D potential already run.
            axis (str): axis along which we want to compute the trapping frequency.
            mf (int or list): Mixed mf state we want to analyze. Default to 0.
            edge_no_surface (float): Position of the edge of the structure. Only needed when no Surface is specified. When a Surface object is given, it is found automatically with the CP masks. Defaults to None.

        Raise:
            TypeError: if only a 2D computation of the potential has been done before plotting.

        Returns:
            float: Trapping frequency along the axis (in Hz)
        """

        if np.ndim(trap_outside) >= 3:
            raise TypeError("The trap given must be one-dimensional")

        trap_pos_index, trap_pos, _, _ = self.get_min_trap(y_outside, trap_outside, edge_no_surface)
        if np.isnan(trap_pos):
            trap_outside3 = np.concatenate((trap_outside, trap_outside, trap_outside))
            y_outside3 = np.concatenate(
                (y_outside - (y_outside[-1] - y_outside[0]), y_outside, y_outside + (y_outside[-1] - y_outside[0]))
            )
            trap_pos_index, trap_pos, _, _ = self.get_min_trap(y_outside3, trap_outside3, edge_no_surface)
            if np.isnan(trap_pos):
                print("[WARNING] No local minimum along the axis. Cannot compute trapping frequency.")
                return 0
            else:
                pass

        try:
            fit = np.polyfit(y_outside[5:], trap_outside[5:], 40)
            pass
        except np.linalg.LinAlgError:
            fit = np.polyfit(y_outside[5:], trap_outside[5:], 20)

        p = np.poly1d(fit)
        der_fit = np.real(np.gradient(p(y_outside), y_outside))
        der2_fit = np.gradient(der_fit, y_outside)
        index_min = np.argmin(np.abs(y_outside - trap_pos))
        moment2 = der2_fit[index_min]
        trap_freq = np.sqrt((moment2 * kB * mK) / (self.simul.atomicsystem.atom.mass)) * (1 / (2 * np.pi))
        return trap_freq

    def get_coord_trap_outside_structure(self, axis, coord1, coord2, mf=0, edge_no_surface=None):
        """Returns the truncation of both the specified axis and the trap along that direction, setting 0 for the coordinate at the edge of the structure.

        Args:
            axis (str): axis along which we are looking at the trap.
            coord1 (float): First coordinate on the orthogonal plane to the
            trapping axis. If axis is Y, coord1 should be the one on X.
            coord2 (float): Second coordinate on the orthogonal plane to the
            trapping axis.
            mf (int or list): Mixed mf state we want to analyze. Default to 0.
            edge_no_surface (float): Position of the edge of the structure. Only needed when no Surface is specified. When a Surface object is given, it is found automatically with the CP masks. Defaults to None.

        Raise:
            TypeError: if only a 2D computation of the potential has been done before plotting.

        Returns:
            (tuple): containing:

                - int: Index of the specified mf state in the array
                - float: Position of the edge of the structure (taken either from the Surface object or given by the user).
                - array: New coordinates, with 0 at the edge of the structure and negative values truncated.
                - array: Corresponding truncation of the trapping potential.
        """
        _, mf = check_mf(self.simul.atomicsystem.f, mf)
        mf_index = int(mf + self.simul.atomicsystem.f)
        coord = set_axis_from_axis(axis, self.simul)
        trap = np.real(self.simul.compute_potential_1D(axis, coord1, coord2))[:, mf_index]

        if axis != self.trapping_axis:
            index_edge = 0
            edge = 0
            y_outside = coord
            trap_outside = trap
        elif axis == self.trapping_axis and type(self.simul.surface).__name__ == "NoSurface":
            if edge_no_surface is None:
                raise ValueError(
                    "No surface for CP interactions have been specified. To restrict the search for the minimum in the right zone, you have to specify an edge"
                )
            edge = edge_no_surface
            index_edge = np.argmin(np.abs(coord - edge))
            y_outside = coord[index_edge:] - edge
            trap_outside = trap[index_edge:]
        else:
            peaks = find_peaks(-self.simul.CP[:, mf_index], height=10)
            if len(peaks[0]) == 0:
                index_edge = 0
                edge = 0
                y_outside = coord
                trap_outside = trap
            elif len(peaks[0]) == 1:
                index_edge = peaks[0][0]
                edge = coord[index_edge - 1]
                y_outside = coord[index_edge:] - edge
                trap_outside = trap[index_edge:]
            elif len(peaks[0]) == 2:
                index_edge = peaks[0]
                edge = coord[index_edge[0] - 1]
                y_outside = coord[index_edge[0] : index_edge[1]] - edge
                trap_outside = trap[index_edge[0] : index_edge[1]]
            else:
                raise ValueError("Too many structures set")

        return mf_index, edge, y_outside, trap_outside

    def ellipticity_plot(self, projection_axis):
        if self.simul.dimension == "2D":
            projection_axis_index = set_axis_index(projection_axis)
            E_amp = np.sqrt(np.sum(abs(self.simul.Etot) ** 2, axis=0))
            E_amp3 = np.stack((E_amp, E_amp, E_amp))
            E_norm = self.simul.Etot / E_amp3
            C = np.cross(E_norm, np.conjugate(E_norm), axisa=0, axisb=0, axisc=0)
            Cz = np.imag(C[projection_axis_index])
            fig, ax = plt.subplots()
            pcm = ax.pcolormesh(
                self.simul.coord1,
                self.simul.coord2,
                np.transpose(Cz),
                cmap="twilight",
                vmin=-1,
                vmax=1,
                shading="gouraud",
            )
            plt.colorbar(pcm)
            plt.title("Ellipticity")

        elif self.simul.dimension == "1D":
            C = np.cross(self.simul.Etot, np.conjugate(self.simul.Etot), axisa=0, axisb=0, axisc=0)
            normal_axis_index = set_axis_index(set_normal_axis(self.simul.plane))
            Cz = np.imag(C[normal_axis_index])
            fig, ax = plt.subplots()
            pcm = plt.plot(self.simul.coord, Cz)
            plt.title("Ellipticity")
        return Cz

    def optimize(self, ymin=0, Pmin1=0, Pmax1=10, Pstep1=1, Pmin2=0, Pmax2=10, Pstep2=1):
        Prange1 = list(reversed(np.arange(Pmin1, Pmax1, Pstep1)))
        Prange2 = np.arange(Pmin2, Pmax2, Pstep2)

        res_pos = np.zeros((len(Prange1), len(Prange2)))
        res_depth = np.zeros((len(Prange1), len(Prange2)))
        res_height = np.zeros((len(Prange1), len(Prange2)))
        res_freq = np.zeros((len(Prange1), len(Prange2)))

        yidx = find_nearest(self.simul.axis, ymin)
        yout = self.simul.axis[yidx:]

        for i, P1 in progressbar_enumerate(Prange1, "\n Optimizing: ", 40):
            for j, P2 in enumerate(Prange2):
                # for i, P1 in enumerate(Prange):
                #     for j, P2 in enumerate(Prange):
                self.simul.trap.set_powers([P1, P2])
                pot = np.real(self.simul.total_potential()[yidx:, 0])
                min_idx, min_pos, depth, height, height_idx = self.get_min_trap(yout, pot)

                ######### frequencies
                if np.isnan(height_idx):
                    trap_freq = 0
                else:
                    height_pos = yout[height_idx]  ## Gives the position of the barrier
                    yleft = min_pos - (min_pos - height_pos) / 2
                    yright = min_pos + (min_pos - height_pos) / 2
                    idx_left = find_nearest(yout, yleft)
                    idx_right = find_nearest(yout, yright)

                    fit = np.polyfit(yout[idx_left:idx_right], pot[idx_left:idx_right], 2)

                    p = np.poly1d(fit)
                    der_fit = np.real(np.gradient(p(yout), yout))
                    der2_fit = np.gradient(der_fit, yout)
                    index_min = np.argmin(np.abs(yout - min_pos))
                    moment2 = der2_fit[index_min]
                    trap_freq = (
                        np.sqrt((moment2 * kB * mK) / (self.simul.atomicsystem.atom.mass)) * (1 / (2 * np.pi)) / kHz
                    )

                ##################

                if abs(depth) > 10:
                    depth = 0
                if abs(height) > 10:
                    height = 0

                depth = abs(depth)
                height = abs(height)
                res_pos[i, j] = min_pos
                res_depth[i, j] = depth
                res_height[i, j] = height
                res_freq[i, j] = trap_freq

        nan_to_zeros(res_pos, res_height, res_depth, res_freq)
        res_pos *= 1e9

        return res_pos, res_depth, res_height, res_freq

    @pyqtSlot()
    def optimize_gui(self, ymin=0, Pmin1=0, Pmax1=10, Pstep1=1, Pmin2=0, Pmax2=10, Pstep2=1):
        Prange1 = list(reversed(np.arange(Pmin1, Pmax1, Pstep1)))
        Prange2 = np.arange(Pmin2, Pmax2, Pstep2)

        res_pos = np.zeros((len(Prange1), len(Prange2)))
        res_depth = np.zeros((len(Prange1), len(Prange2)))
        res_height = np.zeros((len(Prange1), len(Prange2)))
        res_freq = np.zeros((len(Prange1), len(Prange2)))

        yidx = find_nearest(self.simul.axis, ymin)
        yout = self.simul.axis[yidx:]

        t0 = time.time()
        for i, P1 in enumerate(Prange1):
            for j, P2 in enumerate(Prange2):
                # for i, P1 in enumerate(Prange):
                #     for j, P2 in enumerate(Prange):
                self.simul.trap.set_powers([P1, P2])
                pot = np.real(self.simul.total_potential()[yidx:, 0])
                min_idx, min_pos, depth, height, height_idx = self.get_min_trap(yout, pot)

                ######### frequencies
                if np.isnan(height_idx):
                    trap_freq = 0
                else:
                    height_pos = yout[height_idx]  ## Gives the position of the barrier
                    yleft = min_pos - (min_pos - height_pos) / 2
                    yright = min_pos + (min_pos - height_pos) / 2
                    idx_left = find_nearest(yout, yleft)
                    idx_right = find_nearest(yout, yright)

                    fit = np.polyfit(yout[idx_left:idx_right], pot[idx_left:idx_right], 2)

                    p = np.poly1d(fit)
                    der_fit = np.real(np.gradient(p(yout), yout))
                    der2_fit = np.gradient(der_fit, yout)
                    index_min = np.argmin(np.abs(yout - min_pos))
                    moment2 = der2_fit[index_min]
                    trap_freq = (
                        np.sqrt((moment2 * kB * mK) / (self.simul.atomicsystem.atom.mass)) * (1 / (2 * np.pi)) / kHz
                    )

                ##################

                if abs(depth) > 10:
                    depth = 0
                if abs(height) > 10:
                    height = 0

                depth = abs(depth)
                height = abs(height)
                res_pos[i, j] = min_pos
                res_depth[i, j] = depth
                res_height[i, j] = height
                res_freq[i, j] = trap_freq

            self._signal.emit()
            if i == 10:
                self.benchmark.emit(time.time() - t0)
        nan_to_zeros(res_pos, res_height, res_depth, res_freq)
        res_pos *= 1e9
        self.finished.emit()
        self.optim_res = (res_pos, res_depth, res_height, res_freq)
        return res_pos, res_depth, res_height, res_freq

    def optimize_and_show(self, ymin=0, Pmin1=0, Pmax1=10, Pstep1=1, Pmin2=0, Pmax2=10, Pstep2=1):
        ################################################################################################################
        ############################################ Optimization procedure ############################################
        ################################################################################################################

        blockPrint()
        opt_pos, opt_depth, opt_height, opt_freq = self.optimize(
            ymin=ymin, Pmin1=Pmin1, Pmax1=Pmax1, Pstep1=Pstep1, Pmin2=Pmin2, Pmax2=Pmax2, Pstep2=Pstep2
        )
        enablePrint()

        ################################################################################################################
        ################################################## Pretty show #################################################
        ################################################################################################################

        methods = [
            None,
            "none",
            "nearest",
            "bilinear",
            "bicubic",
            "spline16",
            "spline36",
            "hanning",
            "hamming",
            "hermite",
            "kaiser",
            "quadric",
            "catrom",
            "gaussian",
            "bessel",
            "mitchell",
            "sinc",
            "lanczos",
        ]

        color_map = "coolwarm"  # "viridis"  # "gnuplot2"

        Prange1 = list(reversed(np.arange(Pmin1, Pmax1, Pstep1)))
        Prange1 = np.asarray(Prange1)
        Prange2 = np.arange(Pmin2, Pmax2, Pstep2)

        # ax1 = plt.subplot(223)
        # im1 = ax1.imshow(
        #     opt_pos,
        #     cmap=color_map,
        #     interpolation="lanczos",
        #     extent=[Pmin2 / mW, Pmax2 / mW, Pmin1 / mW, Pmax1 / mW],
        #     aspect="auto",
        # )

        # plt.colorbar(im1, ax=ax1)
        # ax1.set_title("Trap position (nm)")
        # ax1.set_xlabel("P2 (mW)")
        # ax1.set_ylabel("P1 (mW)")

        #################################
        # ax4 = plt.subplot(224)
        # im4 = ax4.imshow(
        #     opt_freq,
        #     cmap=color_map,
        #     interpolation="lanczos",
        #     extent=[Pmin2 / mW, Pmax2 / mW, Pmin1 / mW, Pmax1 / mW],
        #     aspect="auto",
        # )
        # # for (j, i), label in np.ndenumerate(opt):
        # #     plt.text(i, j, label, ha="center", va="center")
        # plt.colorbar(im4, ax=ax4)
        # ax4.set_title("Trap frequency (nm)")
        # ax4.set_xlabel("P2 (mW)")
        # ax4.set_ylabel("P1 (mW)")

        # plt.tight_layout()

        ################################

        # ax2 = plt.subplot(221)
        # im2 = ax2.imshow(
        #     opt_depth,
        #     cmap=color_map,
        #     interpolation="lanczos",
        #     extent=[Pmin2 / mW, Pmax2 / mW, Pmin1 / mW, Pmax1 / mW],
        #     aspect="auto",
        # )
        # plt.colorbar(im2, ax=ax2)
        # ax2.set_title("Trap depth (mK)")
        # ax2.set_xlabel("P2 (mW)")
        # ax2.set_ylabel("P1 (mW)")
        # idxs2 = np.unravel_index(opt_depth.argmax(), opt_depth.shape)
        # ax2.plot(
        #     (Prange2[idxs2[1]] + 0.5 * Pstep2) / mW,
        #     (Prange1[idxs2[0]] + 0.5 * Pstep1) / mW,
        #     "o",
        #     color="red",
        #     markersize=12,
        # )

        ##################################

        # ax3 = plt.subplot(211)
        fig = plt.figure(figsize=(3.4, 2.5))  # in inches
        ax3 = fig.add_subplot()  # plt.subplot()
        im3 = ax3.imshow(
            opt_height,
            cmap=color_map,
            interpolation="lanczos",
            extent=[Pmin2 / mW, Pmax2 / mW, Pmin1 / mW, Pmax1 / mW],
            aspect="auto",
            rasterized=True,
        )
        maximas = np.zeros(len(Prange1))
        for i, _ in enumerate(Prange1):
            maximas[i] = Prange2[np.argmax(opt_height[i, :])]
        max_fit = np.polyfit(maximas / mW, Prange1 / mW, 2)
        max_p = np.poly1d(max_fit)
        a = ax3.plot(maximas / mW, max_p(maximas / mW), "--", color="white", lw=3)
        cursor = mplcursors.cursor(
            a,
            highlight=True,  # , highlight_kwargs=_custom_highlight_kwargs, annotation_kwargs=_custom_annotation_kwargs
        )

        @cursor.connect("add")
        def on_add(sel):
            artist = sel.artist
            label = artist.get_label() or ""
            x, y = sel.target
            textx = f"P2 = {x:.1f} nm"
            texty = f"P1 = {y:.2f} mK"
            text = f"{textx}\n{texty}"
            sel.annotation.set_text(text)

        # ax3.plot(maximas / mW, Prange1 / mW, "o", color="green")
        cbar = plt.colorbar(im3, ax=ax3)
        cbar.ax.set_ylabel("Trap depth (mK)", fontsize=8)
        cbar.ax.tick_params(axis="both", which="major", labelsize=8)
        # ax3.set_title("Trap height (mK)")
        ax3.set_xlabel("P2 (mW)", fontsize=8)
        ax3.set_ylabel("P1 (mW)", fontsize=8)
        ax3.tick_params(axis="both", which="major", labelsize=8)
        idxs3 = np.unravel_index(opt_height.argmax(), opt_height.shape)
        # ax3.plot(
        #     (Prange2[idxs3[1]] + 0.5 * Pstep2) / mW,
        #     (Prange1[idxs3[0]] + 0.5 * Pstep1) / mW,
        #     "o",
        #     color="red",
        #     markersize=12,
        # )

        # plt.suptitle(f"Optimal position found : P1 = {Prange1[idxs3[0]]/mW:.2f} mW, P2 = {Prange2[idxs3[1]]/mW:.2f} mW")
        plt.tight_layout(pad=0.2, h_pad=0, w_pad=0.1)
        plt.show()
        fig.savefig("optimizer_figure.pdf", dpi=600)

    @pyqtSlot()
    def optimize_and_show_gui(self, ymin=0, Pmin1=0, Pmax1=10, Pstep1=1, Pmin2=0, Pmax2=10, Pstep2=1):
        ################################################################################################################
        ############################################ Optimization procedure ############################################
        ################################################################################################################

        blockPrint()
        # opt_pos, opt_depth, opt_height, opt_freq = self.optimize_gui(
        #     ymin=ymin, Pmin1=Pmin1, Pmax1=Pmax1, Pstep1=Pstep1, Pmin2=Pmin2, Pmax2=Pmax2, Pstep2=Pstep2
        # )
        opt_pos, opt_depth, opt_height, opt_freq = self.optim_res
        enablePrint()

        ################################################################################################################
        ################################################## Pretty show #################################################
        ################################################################################################################

        methods = [
            None,
            "none",
            "nearest",
            "bilinear",
            "bicubic",
            "spline16",
            "spline36",
            "hanning",
            "hamming",
            "hermite",
            "kaiser",
            "quadric",
            "catrom",
            "gaussian",
            "bessel",
            "mitchell",
            "sinc",
            "lanczos",
        ]

        # Prange1 = np.arange(Pmin1, Pmax1, Pstep1)
        Prange1 = list(reversed(np.arange(Pmin1, Pmax1, Pstep1)))
        Prange1 = np.asarray(Prange1)
        Prange2 = np.arange(Pmin2, Pmax2, Pstep2)

        # ax1 = plt.subplot(212)
        ax1 = plt.subplot(223)
        im1 = ax1.imshow(
            opt_pos,
            cmap="gnuplot2",
            interpolation="lanczos",
            extent=[Pmin2 / mW, Pmax2 / mW, Pmin1 / mW, Pmax1 / mW],
            aspect="auto",
        )
        # for (j, i), label in np.ndenumerate(opt):
        #     plt.text(i, j, label, ha="center", va="center")
        plt.colorbar(im1, ax=ax1)
        ax1.set_title("Trap position (nm)")
        ax1.set_xlabel("P2 (mW)")
        ax1.set_ylabel("P1 (mW)")

        #################################
        ax4 = plt.subplot(224)
        im4 = ax4.imshow(
            opt_freq,
            cmap="gnuplot2",
            interpolation="lanczos",
            extent=[Pmin2 / mW, Pmax2 / mW, Pmin1 / mW, Pmax1 / mW],
            aspect="auto",
        )
        # for (j, i), label in np.ndenumerate(opt):
        #     plt.text(i, j, label, ha="center", va="center")
        plt.colorbar(im4, ax=ax4)
        ax4.set_title("Trap frequency (nm)")
        ax4.set_xlabel("P2 (mW)")
        ax4.set_ylabel("P1 (mW)")

        plt.tight_layout()

        ################################

        # ax2 = plt.subplot(221)
        # im2 = ax2.imshow(
        #     opt_depth,
        #     cmap="gnuplot2",
        #     interpolation="lanczos",
        #     extent=[Pmin2 / mW, Pmax2 / mW, Pmin1 / mW, Pmax1 / mW],
        #     aspect="auto",
        # )
        # plt.colorbar(im2, ax=ax2)
        # ax2.set_title("Trap depth (mK)")
        # ax2.set_xlabel("P2 (mW)")
        # ax2.set_ylabel("P1 (mW)")
        # idxs2 = np.unravel_index(opt_depth.argmax(), opt_depth.shape)
        # ax2.plot(
        #     (Prange2[idxs2[1]] + 0.5 * Pstep2) / mW,
        #     (Prange1[idxs2[0]] + 0.5 * Pstep1) / mW,
        #     "o",
        #     color="red",
        #     markersize=12,
        # )

        ##################################

        # ax3 = plt.subplot(222)
        ax3 = plt.subplot(211)
        im3 = ax3.imshow(
            opt_height,
            cmap="gnuplot2",
            interpolation="lanczos",
            extent=[Pmin2 / mW, Pmax2 / mW, Pmin1 / mW, Pmax1 / mW],
            aspect="auto",
        )
        maximas = np.zeros(len(Prange1))
        for i, _ in enumerate(Prange1):
            maximas[i] = Prange2[np.argmax(opt_height[i, :])]
        max_fit = np.polyfit(maximas / mW, Prange1 / mW, 2)
        max_p = np.poly1d(max_fit)
        a = ax3.plot(maximas / mW, max_p(maximas / mW), "--", color="white", lw=3)
        cursor = mplcursors.cursor(
            a,
            highlight=True,  # , highlight_kwargs=_custom_highlight_kwargs, annotation_kwargs=_custom_annotation_kwargs
        )
        # ax3.plot(maximas / mW, Prange1 / mW, "o", color="green")
        plt.colorbar(im3, ax=ax3)
        ax3.set_title("Trap height (mK)")
        ax3.set_xlabel("P2 (mW)")
        ax3.set_ylabel("P1 (mW)")
        idxs3 = np.unravel_index(opt_height.argmax(), opt_height.shape)
        ax3.plot(
            (Prange2[idxs3[1]] + 0.5 * Pstep2) / mW,
            (Prange1[idxs3[0]] + 0.5 * Pstep1) / mW,
            "o",
            color="red",
            markersize=12,
        )

        plt.suptitle(f"Optimal position found : P1 = {Prange1[idxs3[0]]/mW:.2f} mW, P2 = {Prange2[idxs3[1]]/mW:.2f} mW")
        plt.tight_layout()
        plt.show()


class DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""

    def __init__(self, *args, **kwargs):
        self.allowed_vals = kwargs.pop("allowed_vals", None)
        self.previous_val = kwargs["valinit"]
        Slider.__init__(self, *args, **kwargs)
        if self.allowed_vals.all() == None:
            self.allowed_vals = [self.valmin, self.valmax]
        for k in range(len(self.allowed_vals)):
            if self.orientation == "vertical":
                self.hline = self.ax.axhline(self.allowed_vals[k], 0, 1, color="r", lw=1)
            else:
                self.vline = self.ax.axvline(self.allowed_vals[k], 0, 1, color="r", lw=1)

    def set_val(self, val):
        discrete_val = self.allowed_vals[abs(val - self.allowed_vals).argmin()]
        val = discrete_val
        xy = self.poly.xy
        if self.orientation == "vertical":
            xy[1] = 0, val
            xy[2] = 1, val
        else:
            xy[2] = val, 1
            xy[3] = val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % val)
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.items():
            func(val)