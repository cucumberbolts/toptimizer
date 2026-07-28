"""
Implementation of BESO optimization algorithm

July 2026
Julian Poon
"""

from dataclasses import dataclass
from enum import Flag

from typing import Iterable, Generator

import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Fix(Flag):
    """ Flag to specify which component of a node to fix """
    X  = 0b01
    Y  = 0b10
    XY = 0b11


class Design:
    """
    Class for specifying the boundary conditions for the BESO optimizer
    """

    def __init__(self, nelx: int, nely: int):
        self.nelx = nelx
        self.nely = nely
        self.forces = None
        self.fixed = set()
        # self.passive = {}  TODO: Implement this


    def add_forces(self, nodes_x: Iterable[int], nodes_y: Iterable[int], *, forces_x: Iterable[float] = None, forces_y: Iterable[float] = None) -> None:
        """
        Add loading forces. If either forces_x or forces_y
        is shorter than the other or is None, then zeros
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

        forces = {}
        
        # Map each force value to their corresponding degree of freedom on which they act
        num_forces = len(forces_x)  # The lengths of all four lists should be the same at this point
        if num_forces != 1: raise ValueError("Loading with more than one force is not supported (yet!)")
        for i in range(num_forces):
            if forces_x[i] != 0:
                dof = 2 * (nodes_y[i] * (self.nelx + 1) + nodes_x[i])
                # TODO: Think about whether or not the force should be added (rather
                # than overwritten) in the case that there is an existing force acting
                # on that node specified by a previous call to this function
                forces[dof] = forces_x[i]

            if forces_y[i] != 0:
                dof = 2 * (nodes_y[i] * (self.nelx + 1) + nodes_x[i]) + 1
                forces[dof] = forces_y[i]


        dof_count = 2 * (self.nelx + 1) * (self.nely + 1)

        self.forces = csc_array(
            (
                list(forces.values()),
                (
                    list(forces.keys()),  # Row of matrix
                    range(num_forces),    # Column of matrix
                )
            ),
            shape=(dof_count, num_forces)
        )


    def add_fixed(self, fix: Fix, nodes_x: Iterable[int], nodes_y: Iterable[int]) -> None:
        """
        Add fixed degrees of freedom

        Parameters:
        -----------
        fix: Fix
            Fix the nodes in x, y, or both
        nodes_x: Iterable[int]
            x coordinates of the nodes to fix
        nodes_y: Iterable[int]
            y coordinates of the nodes to fix
        """

        if len(nodes_x) != len(nodes_y):
            raise ValueError("Number of fixed node component lists do not match")

        for node_x, node_y in zip(nodes_x, nodes_y):
            if fix & Fix.X:
                self.fixed.add(2 * (node_y * (self.nelx + 1) + node_x))
            if fix & Fix.Y:
                self.fixed.add(2 * (node_y * (self.nelx + 1) + node_x) + 1)


@dataclass(kw_only=True, frozen=True)
class BESOParams:
    """ Class for defining the parameters of BESO """
    penal: float   = 3.0   # Penalization exponent for FEA
    volfrac: float = 0.3   # Volume fraction of the optimized design
    rmin: float    = 1.5   # Sensitivity filter radius
    ert: float     = 0.02  # Evolutionary rate of BESO


# TODO: Figure out how to refactor this
E       = 1     # Young's modulus
NU      = 0.3   # Poisson's ratio


# Coefficients for the stiffness matrix
k = np.array([
    1/2-NU/6,   1/8+NU/8, -1/4-NU/12, -1/8+3*NU/8,
   -1/4+NU/12, -1/8-NU/8,       NU/6,  1/8-3*NU/8,
])

# Local stiffness matrix for each element
K_LOCAL = E/(1 - NU**2) * np.array([
    [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
])


SOLID = 1.0
VOID = 0.001


def fea(x: np.array, penal: float) -> np.array:
    """
    Perform finite element analysis
    to obtain displacement vector

    Parameters
    ----------
    x     : np.array
        2D material density distribution for each element
    penal : float
        Penalization factor (usually 3)

    Returns
    -------
    np.array
        The global displacement vector
    """

    nely, nelx = x.shape

    dof_count = 2 * (nelx + 1) * (nely + 1)

    # Global stiffness matrix
    # K = dok_array((dof_count, dof_count))
    K = np.zeros((dof_count, dof_count))
    U = np.zeros(dof_count)

    # Assemble the global stiffness matrix
    for ely in range(nely):
        for elx in range(nelx):
            # Upper left and lower left element node number in global node matrix
            ul = elx + (nelx + 1) * ely
            ll = elx + (nelx + 1) * (ely + 1)

            edof = [
                2*ul,     2*ul+1,      # upper-left
                2*(ul+1), 2*(ul+1)+1,  # upper-right
                2*(ll+1), 2*(ll+1)+1,  # lower-right
                2*ll,     2*ll+1,      # lower-left
            ]

            K[np.ix_(edof, edof)] += (x[ely, elx] ** penal) * K_LOCAL

    free_dofs = [d for d in range(dof_count) if d not in design.fixed]

    U[np.ix_(free_dofs)] = spsolve(
        K[np.ix_(free_dofs, free_dofs)],
        design.forces[np.ix_(free_dofs)]
    )

    U[np.ix_(list(design.fixed))] = 0

    return U


def sensitivity(x: np.array, U: np.array, params: BESOParams) -> np.array:
    """
    Calculate the sensitivities of each element
    dc = 1/2 * Ui^T * Ki * Ui

    Parameters
    ----------
    x : np.array
        Array of design variables, 'SOLID' for solid, 'VOID' for void
    U : np.array
        Local displacements

    Returns
    -------
    np.array
        Array of sensitivity values for each element in the design domain
    """

    nely, nelx = x.shape

    c = 0.0
    dc = np.zeros(x.shape)

    for ely in range(nely):
        for elx in range(nelx):
            # Upper left and lower left element node number in global node matrix
            ul = elx + (nelx + 1) * ely
            ll = elx + (nelx + 1) * (ely + 1)

            edof = [
                2*ul,     2*ul+1,      # upper-left
                2*(ul+1), 2*(ul+1)+1,  # upper-right
                2*(ll+1), 2*(ll+1)+1,  # lower-right
                2*ll,     2*ll+1,      # lower-left
            ]

            # Get the element displacement from the global displacement vector
            ue = U[np.ix_(edof)]

            # Calculate the compliance
            # c = 0.5 * ue.T @ K_LOCAL @ ue
            c = ue.T @ K_LOCAL @ ue

            # Calculate the sensitivity
            # dc[ely, elx] = -params.penal * x[ely, elx] ** (params.penal - 1) * c
            dc[ely, elx] = x[ely, elx] ** (params.penal - 1) * c

    # Filter the sensitivities
    dcf = filter(x, params.rmin, dc)

    return dcf


def filter(x: np.array, rmin: float, dc: np.array) -> np.array:
    """
    Filters the sensitivities

    Parameters
    ----------
    x    : np.array
        Array of design variables, 'SOLID' for solid, 'VOID' for void
    U    : np.array
        Local displacements
    rmin : float
        Filter radius
    dc   : np.array
        Array of sensitivities

    Returns
    -------
    np.array
        The filtered sensitivities
    """

    nely, nelx = dc.shape

    f = int(np.floor(rmin))

    # The filtered sensitivities
    dcn = np.zeros(dc.shape)

    for j in range(nely):
        for i in range(nelx):
            sum = 0.0
            for l in range(max(j - f, 0), min(j + f + 1, nely)):
                for k in range(max(i - f, 0), min(i + f + 1, nelx)):
                    fac = rmin - np.sqrt((i - k)**2 + (j - l)**2)

                    sum += max(0, fac)

                    # dcn[j, i] += max(0, fac) * x[l, k] * dc[l, k]
                    dcn[j, i] += max(0, fac) * dc[l, k]

            # dcn[j, i] /= x[j, i] * sum
            dcn[j, i] /= sum

    return dcn


def update(x: np.array, dc: np.array, vol: float) -> np.array:
    """
    Updates the design variables given filtered sensitivities
    and target volume fraction using the bisection algorithm.

    Parameters
    ----------
    x   : np.array
        Array of design variables, 'SOLID' for solid, 'VOID' for void
    dc  : np.array
        Array of filtered sensitivities
    vol : float
        Target volume

    Returns
    -------
    np.array
        The updated design variable values
    """

    nely, nelx = x.shape

    # Sensitivity threshold:
    # Variables with sensitivities lower than the threshold are made solid and
    # variables with sensitivities higher than the threshold are made void.
    threshold = 0.0

    # Find the sensitivity threshold with the bisection algorithm

    low  = dc.min()
    high = dc.max()

    while (high - low) / high > 0.00001:
        threshold = (low + high) * 0.5

        # FROM BESO2D.py
        # x = np.maximum(np.tile(VOID, x.shape), np.sign(dc - threshold))

        for ely in range(nely):
            for elx in range(nelx):
                if dc[ely, elx] > threshold:
                    x[ely, elx] = SOLID
                else:
                    x[ely, elx] = VOID

        if x.sum() > vol:
            low = threshold
        else:
            high = threshold

    return x


def BESO(design: Design, params: BESOParams = None) -> Generator[np.array, None, None]:
    """
    Optimize topology using the BESO algorithm and
    yield the design variable values for each iteration
    """

    if params is None:
        params = BESOParams()

    # Initialize iteration
    x  = np.tile(SOLID, (design.nely, design.nelx))  # Design variables
    dc = np.zeros((design.nely, design.nelx))        # Sensitivity values

    current_vol = design.nelx * design.nely * SOLID
    target_vol  = design.nelx * design.nely * params.volfrac

    loop = 0
    change = 0.1

    # Start Iteration
    while change > 0.01:
        x_old = np.copy(x)

        # Finite element analysis
        U = fea(x, params.penal)

        # Sensitivity analysis
        dc_old = np.copy(dc)
        dc = sensitivity(x, U, params)
        # Average the sensitivities with the previous iteration
        if loop > 0:
            dc = 0.5 * (dc + dc_old)

        # Update the current iteration's target volume
        # according to the evolutionary rate
        if current_vol > target_vol:
            current_vol *= 1 - params.ert

        # Change the design variables according to sensitivity analysis
        x = update(x, dc, current_vol)

        change = abs((x - x_old).sum() / x.sum())

        # print(f"It.: {loop} Obj.: {c:10.4f} Vol.: {x.sum() / nelx / nely:6.3f} Ch.: {change:6.3f}")
        print(f"It.: {loop} Vol.: {x.sum() / (design.nelx * design.nely):6.3f} Ch.: {change:6.3f}")

        loop += 1

        yield x


def anim_update(x: np.array, image: matplotlib.image.AxesImage) -> list[matplotlib.artist.Artist]:
    """
    Redraws the image when called by FuncAnimation

    Parameters:
    -----------
    x     : np.array
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


if __name__ == "__main__":
    design = Design(60, 20)

    # Downwards force on the top-left corner
    design.add_forces([0], [0], forces_y=[-1])

    # Fix the bottom-right corner in the y direction
    design.add_fixed(Fix.Y, [design.nelx], [design.nely])
    # Fix the left wall in the x direction
    design.add_fixed(Fix.X,
        [0] * (design.nely + 1),
        range(design.nely + 1),
    )

    optimizer = BESO(design)

    fig, ax = plt.subplots()
    im = ax.matshow(np.zeros((design.nely, design.nelx)))

    # Specify the upper and lower bounds of the values to
    # be plotted so the colours can be displayed properly
    im.set_clim(0.0, 1.0)

    anim = FuncAnimation(
        fig,
        anim_update,
        fargs=[im],
        frames=optimizer,
        repeat_delay=0.0,
        save_count=10,
        blit=True,
        repeat=False
    )

    plt.show()
