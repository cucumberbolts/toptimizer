"""
Common funcionalities for topology optimization
"""

from enum import IntEnum, Flag

from typing import Iterable, Generator

import numpy as np
from scipy.sparse import coo_array, csc_array

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Fix(Flag):
    """ Flag to specify which component of a node to fix """
    X  = 0b01
    Y  = 0b10
    XY = 0b11


class Passive(IntEnum):
    """ Enum to specify passive elements """
    NONE = 0
    VOID = 1
    SOLID = 2


class Design:
    """
    Class for specifying the boundary conditions for the BESO optimizer
    """

    def __init__(self, nelx: int, nely: int):
        self.nelx = nelx
        self.nely = nely
        self.__forces_dict = {} # Used for the construction of the self.forces matrix
        self.__forces = None
        self.fixed = set()
        self.passive = np.full((nely, nelx), Passive.NONE)


    def add_forces(self, nodes_x: Iterable[int], nodes_y: Iterable[int], *, forces_x: Iterable[float] = None, forces_y: Iterable[float] = None) -> None:
        """
        Add loading forces. If either forces_x or forces_y
        is shorter than the other, or is None, then zeros
        will be added to the end to match the other's length.

        Parameters:
        -----------
        nodes_x: Iterable[int]
            x components of the force nodes
        nodes_y: Iterable[int]
            y components of the force nodes
        forces_x: Iterable[float]
            x components of the loading forces
        forces_y: Iterable[float]
            y components of the loading forces
        """

        if forces_y is None:
            forces_y = [0] * len(forces_x)
        elif forces_x is None:
            forces_x = [0] * len(forces_y)

        # Fill-in missing force components with zeros
        if len(forces_x) < len(forces_y):
            forces_x += [0] * (forces_y - forces_x)
        elif len(forces_y) < len(forces_x):
            forces_y += [0] * (forces_x - forces_y)

        if len(nodes_x) != len(nodes_y):
            raise ValueError("Number of force coordinate lists do not match")

        if len(nodes_x) != len(forces_x):
            raise ValueError("Number of force coordinates and force values do not match")

        # Map each force value to their corresponding degree of freedom on which they act
        num_forces = len(forces_x)  # The lengths of all four lists should be the same at this point
        for i in range(num_forces):
            if forces_x[i] != 0:
                dof = 2 * (nodes_y[i] * (self.nelx + 1) + nodes_x[i])
                # TODO: Think about whether or not the force should be added (rather
                # than overwritten) in the case that there is an existing force acting
                # on that node specified by a previous call to this function
                self.__forces_dict[dof] = forces_x[i]

            if forces_y[i] != 0:
                dof = 2 * (nodes_y[i] * (self.nelx + 1) + nodes_x[i]) + 1
                self.__forces_dict[dof] = forces_y[i]


        # Reset the force matrix so we know the data has
        # been updated and should be reconstructed later
        self.__forces = None


    def get_forces(self) -> np.array:
        """
        Assembles the force matrix into a csc_array and returns it

        Returns:
        --------
        np.array:
            The force matrix
        """

        if self.__forces is None:
            dof_count = 2 * (self.nelx + 1) * (self.nely + 1)
            num_forces = len(self.__forces_dict)

            self.__forces = csc_array(
                (
                    list(self.__forces_dict.values()),
                    (
                        list(self.__forces_dict.keys()),  # Row of matrix
                        range(num_forces),  # Column of matrix
                    )
                ),
                shape=(dof_count, num_forces)
            )

        return self.__forces


    def add_fixed(self, fix: Fix, nodes_x: Iterable[int], nodes_y: Iterable[int]) -> None:
        """
        Define fixed degrees of freedom

        Parameters:
        -----------
        fix : Fix
            Fix the nodes in x, y, or both
        nodes_x : Iterable[int]
            x coordinates of the nodes to fix
        nodes_y : Iterable[int]
            y coordinates of the nodes to fix
        """

        if len(nodes_x) != len(nodes_y):
            raise ValueError("Number of fixed node coordinate lists do not match")

        for node_x, node_y in zip(nodes_x, nodes_y):
            if fix & Fix.X:
                self.fixed.add(2 * (node_y * (self.nelx + 1) + node_x))
            if fix & Fix.Y:
                self.fixed.add(2 * (node_y * (self.nelx + 1) + node_x) + 1)


    def add_passive(self, passive: Passive, elems_x: Iterable[int], elems_y: Iterable[int]) -> None:
        """
        Define passive elements

        Parameters:
        -----------
        passive : Passive
            Force the elements to be void or solid
        elems_x : Iterable[int]
            x coordinates of passive elements
        elems_y : Iterable[int]
            y coordinates of passive elements
        """

        if len(elems_x) != len(elems_y):
            raise ValueError("Number of passive element coordinate lists do not match")

        self.passive[np.ix_(elems_y, elems_x)] = passive


def animate(design: Design, optimizer: Generator[np.array, None, None]) -> None:
    """
    Create an animation in matplotlib of the optimization algorithm

    Parameters:
    -----------
    optimizer : Generator[np.array, None, None]
        The generator object that yields the design variables
    """

    fig, ax = plt.subplots()
    im = ax.matshow(np.zeros((design.nely, design.nelx)))

    # Specify the upper and lower bounds of the values to
    # be plotted so the colours can be displayed properly
    im.set_clim(0.0, 1.0)

    def anim_update(x: np.array, image: matplotlib.image.AxesImage) -> list[matplotlib.artist.Artist]:
        """
        Redraws the image when called by FuncAnimation

        Parameters:
        -----------
        x : np.array
            The design variable values
        image : matplotlib.image.AxesImage
            The Axes image returned by plt.matshow()

        Returns:
        --------
        list[matplotlib.artist.Artist]:
            The Artist objects whose data were updated
        """

        image.set_data(x)
        return [image]

    anim = FuncAnimation(
        fig,
        anim_update,
        fargs=[im],
        frames=optimizer,
        repeat_delay=0.0,
        save_count=10,
        interval=0,
        blit=True,
        repeat=False
    )

    plt.show()
