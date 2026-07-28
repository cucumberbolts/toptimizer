"""
Implementation of BESO optimization algorithm

July 2026
Julian Poon
"""

import numpy as np
from scipy.sparse import csr_array, dok_array
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist

# DESIGN PARAMETERS
NELX    = 60
NELY    = 20
PENAL   = 3.0
RMIN    = 1.5
VOLFRAC = 0.3
ERT     = 0.02

# Material Properties
E = 1       # Young's modulus
NU = 0.3    # Poisson's ratio

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

# Define loading forces
# csr_array((data, (row_ind, col_ind)),[shape=(M,N)])
F = csr_array(
    (
        [-1],
        ([1], [0])
    ),
    shape=[2 * (NELX + 1) * (NELY + 1), 1]
)


SOLID = 1.0
VOID = 0.001


def fea(x, penal) -> np.array:
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

    # Horizontal constraints on left wall
    fixed_dofs = list(range(0, dof_count, 2 * (nelx + 1))) + [dof_count - 1]

    free_dofs = [d for d in range(dof_count) if d not in fixed_dofs]

    U[np.ix_(free_dofs)] = spsolve(
        K[np.ix_(free_dofs, free_dofs)],
        F[np.ix_(free_dofs)]
    )

    U[np.ix_(fixed_dofs)] = 0

    return U


def sensitivity(x: np.array, U: np.array) -> np.array:
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
            # dc[ely, elx] = -PENAL * x[ely, elx] ** (PENAL - 1) * c
            dc[ely, elx] = x[ely, elx] ** (PENAL - 1) * c

    # Filter the sensitivities
    dcf = filter(x, RMIN, dc)

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


def BESO():
    """
    Optimize topology using the BESO algorithm and
    yield the design variable values for each iteration
    """

    # Initialize iteration
    x  = np.tile(SOLID, (NELY, NELX))  # Design variables
    dc = np.zeros((NELY, NELX))        # Sensitivity values

    current_vol = NELX * NELY * SOLID
    target_vol  = NELX * NELY * VOLFRAC

    loop = 0
    change = 0.1

    # Start Iteration
    while change > 0.01:
        x_old = np.copy(x)

        # Finite element analysis
        U = fea(x, PENAL)

        # Sensitivity analysis
        dc_old = np.copy(dc)
        dc = sensitivity(x, U)
        # Average the sensitivities with the previous iteration
        if loop > 0:
            dc = 0.5 * (dc + dc_old)

        # Update the current iteration's target volume
        # according to the evolutionary rate
        if current_vol > target_vol:
            current_vol *= 1 - ERT

        # Change the design variables according to sensitivity analysis
        x = update(x, dc, current_vol)

        change = abs((x - x_old).sum() / x.sum())

        # print(f"It.: {loop} Obj.: {c:10.4f} Vol.: {x.sum() / nelx / nely:6.3f} Ch.: {change:6.3f}")
        print(f"It.: {loop} Vol.: {x.sum() / (NELX * NELY):6.3f} Ch.: {change:6.3f}")

        loop += 1

        yield x


def anim_update(x: np.array, image):
    """
    Redraws the image when called by FuncAnimation

    Parameters:
    -----------
    x     : np.array
        The design variable values
    image :
        The Axes image returned by plt.matshow()

    Returns:
    --------
    list[matplotlib.artist.Artist]:
        The Artist objects whose data were updated
    """

    image.set_data(x)
    return [image]


if __name__ == "__main__":
    optimizer = BESO()

    fig, ax = plt.subplots()
    im = ax.matshow(np.zeros((NELY, NELX)))

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
