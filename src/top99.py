"""
A basic topology optimizer translated from Ole
Sigmund's 99 line topology optimization code
written in Matlab

Sigmund, O. (2001). A 99 line topology optimization
    code written in Matlab. In Struct Multidisc Optim
    (Vol. 21). Springer-Verlag. http://www.topopt.dtu.dk.

This page was also referenced during translation:
https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

Julian Poon May 30, 2026
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist

import argparse

from pyinstrument import Profiler

# Topology optimizer parameters
# See the top() function for details
NELX = 60
NELY = 20
VOLFRAC = 0.3
PENAL = 3
RMIN = 1.5


def local_stiffness() -> np.array:
    """
    Compute the stiffness matrix for a single element

    Returns
    -------
    np.array
        The local stiffness matrix for each element
    """

    E = 1    # Young's modulus
    v = 0.3  # Poisson's ratio (typically 0.3)

    # Coefficients for the stiffness matrix
    k = np.array([
        1/2-v/6,   1/8+v/8, -1/4-v/12, -1/8+3*v/8,
       -1/4+v/12, -1/8-v/8,       v/6,  1/8-3*v/8,
    ])

    KE = E/(1 - v**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
    ])

    return KE


def fea(nelx, nely, x, penal) -> np.array:
    """
    Perform finite element analysis
    to obtain displacement vector

    Parameters
    ----------
    nelx  : int
        Number of elements in x
    nely  : int
        Number of elements in y
    x     : np.array
        2D material density distribution for each element
    penal : float
        Penalization factor (usually 3)

    Returns
    -------
    np.array
        The global displacement vector
    """

    KE = local_stiffness()

    dof_count = 2 * (nelx + 1) * (nely + 1)

    # Use sparse matrices?
    # Global stiffness matrix
    K = np.zeros((dof_count, dof_count))
    # Global force vector
    F = np.zeros(dof_count)
    # Global displacement vector
    U = np.zeros(dof_count)

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

            dK = (x[ely, elx] ** penal) * KE
            K[np.ix_(edof, edof)] = K[np.ix_(edof, edof)] + dK

    #############################
    # Define Loads and Supports #
    #############################

    # Downwards load on top left
    F[1] = -1
    # Horizontal constraints on left wall
    fixed_dofs = list(range(0, dof_count, 2 * (nelx + 1))) + [dof_count - 1]

    """
    # Downwards load on top right
    F[2 * (nelx + 1) - 1] = -1
    # Horizontal constraints on right wall
    fixed_dofs = list(range(2 * (nelx + 1) - 2, dof_count, 2 * (nelx + 1)))\
        + [dof_count - 2 * (nelx + 1) + 1]

    # Downwards distributed load on top
    # for i in range(1, 2 * (nelx + 1), 2):
    #     F[i] = -10
    # F[dof_count - 1] = 10

    # Downwards force on the top middle
    F[nelx + 1] = -1
    # Pin supports on bottom left and right
    fixed_dofs = [dof_count - 1] + [dof_count - 2 * (nelx + 1) + 1]

    # Fixed
    # F[dof_count - 1] = 1
    # F[2 * (nelx + 1) - 2] = -1
    # fixed_dofs = [dof_count - 2 * (nelx + 1) + 1] + [nelx + 1] + [dof_count - 2 * (nelx + 1)]
    """

    free_dofs = [d for d in range(dof_count) if d not in fixed_dofs]

    U[np.ix_(free_dofs)] = np.linalg.solve(
        K[np.ix_(free_dofs, free_dofs)],
        F[np.ix_(free_dofs)]
    )

    U[np.ix_(fixed_dofs)] = 0

    return U


def check(nelx, nely, rmin, x, dc) -> np.array:
    """
    Filter sensitivities by applying the mesh-independency filter.
    This is done to avoid the "checkerboarding problem"

    Parameters:
    -----------
    nelx:
        Number of elements in x
    nely:
        Number of elements in y
    rmin:
        Filter radius that determines the extent of
        the neighborhood for filtering. It defines
        how far the filter will reach to neighboring
        elements to smooth out the sensitivities.
    x:
        Design variable with current material density distribution
    dc:
        Sensitivity of the compliance (objective function) with respect to the design variables x

    Returns:
    --------
    np.array:
        The filtered sensitivities. These are modified
        versions of the original sensitivities dc after
        appllying the mesh-independency filter.
    """

    dcn = np.zeros((nely, nelx))

    for j in range(nely):
        for i in range(nelx):
            # Accumulates the sum of the weight factors fac
            # used in the filtering process for each element
            # filtering neighbouring elements within the radius
            # rmin these loops iterate over neughbouring elements
            # within a square filter of size 2*rmin around
            # the current element (i,j)
            sum = 0.0

            # Loops through each element within 'rmin'
            # radius of the current element

            # k: the column index of a neighbouring element
            # l: the row index of a neighbouring element
            f = int(np.floor(rmin))
            for l in range(max(j - f, 0), min(j + f + 1, nely)):
                for k in range(max(i - f, 0), min(i + f + 1, nelx)):
                    # Weight factor decreases as the distance increases.
                    # Elements closer to the current element (i,j) have
                    # a higher weight in the filtering process. This
                    # ensures that elements within the filter radius rmin
                    # have a positive influence and elements beyond the
                    # radius have no influence
                    fac = rmin - np.sqrt((i - k)**2 + (j - l)**2)

                    # Elements beyond rmin should have no influence
                    sum += max(0, fac)

                    # Adds the weighted sensitivity of the neighboring
                    # element (l, k) to the filtered sensitivity dcn(j, i)
                    # of the current element (i, j). The weight factor fac
                    # is multiplied by the density x(l,k) and the original
                    # sensitivity dc(l,k) of the neighboring element.
                    dcn[j, i] += max(0, fac) * x[l, k] * dc[l, k]

            # After all the neighboring elements' contributions
            # have been added, the filtered sensitivity dcn(j, i)
            # is normalized by dividing it by the total sum of
            # the weight factors.
            dcn[j, i] /= x[j, i] * sum

    return dcn


def OC(nelx, nely, x, volfrac, dc) -> np.array:
    """
    Optimality criteria update

    Parameters:
    -----------
    nelx:
        Number of elements in x
    nely:
        Number of elements in y
    x:
        Design variable with current material density distribution
    volfrac:
        The amount of material in the design domain to use
    dc:
        Sensitivity of the compliance (objective function) with respect to the design variables x

    Returns:
    --------
    np.array:
        The new design variables
    """

    # Upper and lower bound for Lagrange multiplier
    # Bounds are used in bisectioning method to find it
    l_low  = 0
    l_high = 100000

    # The move limit, which restricts how much the
    # material densities can change between iterations.
    # This ensures stability and prevents large, abrupt
    # changes in the design
    move = 0.2

    x_new = np.zeros((nely, nelx))

    # Set to 1 for void, 2 for solid
    # passive = np.zeros((nely, nelx))

    # Set the top line to solid
    # for i in range(nelx):
    #     passive[0, i] = 2

    # print(passive)

    # Bisection method to find Lagrange multiplier
    while l_high - l_low > 0.0001:
        l_mid = 0.5 * (l_low + l_high)

        x_new = np.maximum(0.001, np.maximum(x - move, np.minimum(1, np.minimum(x + move, x * np.sqrt(-dc / l_mid)))))

        if np.sum(x_new) - volfrac * nelx * nely > 0:
            l_low = l_mid
        else:
            l_high = l_mid

    return x_new


def top(nelx, nely, volfrac, penal, rmin) -> np.array:
    """
    The topology optimizing function

    Parameters:
    -----------
    nelx:
        Number of elements in x
    nely:
        Number of elements in y
    volfrac:
        The fraction of the total volume of the design
        domain that the final design should occupy
    penal:
        Penalization factor (usually 3)
    rmin:
        Filter radius
    """
    x = np.full((nely, nelx), volfrac)

    loop = 0    # Counter to track number of optimization iterations
    change = 1  # Variable to store maximum change in material distribution between iterations

    # START ITERATION
    # optimization stops when the change in
    # material distribution is less than 1%

    while change > 0.01:
        loop += 1
        xold = np.copy(x)

        U = fea(nelx, nely, x, penal)

        KE = local_stiffness()

        # Initialize compliance (objective function) to zero
        c = 0.0

        dc = np.zeros((nely, nelx))

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

                # Extracts the element displacement vector Ue
                # from the global displacement vector U.
                # This is needed to compute the element's contribution
                # to the compliance and its sensitivity
                Ue = U[np.ix_(edof)]
                # Compliance calculation
                t = Ue.T @ KE @ Ue
                c += (x[ely, elx]**penal) * Ue.T @ KE @ Ue    # Accumulates the total compliance
                dc[ely, elx] = -penal * x[ely, elx]**(penal-1) * t  # Sensitivity of compliance with respect to material density calculation

        # Filtering of sensitivities
        dc = check(nelx, nely, rmin, x, dc)

        # Design update by the optimality criteria method
        x = OC(nelx, nely, x, volfrac, dc)

        # change = max(max(abs(x - xold)))
        change = abs(x - xold).max()

        print(f"It.: {loop} Obj.: {c:10.4f} Vol.: {x.sum() / nelx / nely:6.3f} Ch.: {change:6.3f}")

        yield x


def anim_update(x, image) -> list[Artist]:
    """
    Function that redraws the image
    when called by FuncAnimation

    Parameters:
    -----------
    x:
        The density values of each element in the design domain
    image:
        The AxesImage returned by plt.matshow()

    Returns:
    --------
    list[matplotlib.artist.Artist]:
        The Artist objects whose data were updated
    """

    image.set_data(x)
    return [image]


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns:
    --------
    argparse.Namespace:
        An object containing all the options
        specified in the command line
    """
    parser = argparse.ArgumentParser(description="A basic topology optimizer")

    # Setting for benchmarking the topology
    # optimization code which will only show
    # the final result
    parser.add_argument(
        "-p", "--profile",
        action="store_true",
        help="Enable benchmarking mode"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    optimizer = top(NELX, NELY, VOLFRAC, PENAL, RMIN)

    fig, ax = plt.subplots()
    im = ax.matshow(np.zeros((NELY, NELX)))

    # Specify the upper and lower bounds of the values to
    # be plotted so the colours can be displayed properly
    im.set_clim(0.0, 1.0)

    if args.profile:
        final_image = None

        with Profiler(interval=0.05) as profiler:
            for x in optimizer:
                final_image = x

        profiler.print()

        ax.matshow(final_image)
    else:
        anim = FuncAnimation(
            fig,
            anim_update,
            fargs=[im],
            frames=optimizer,
            repeat_delay=0,  # Do not add a delay between frames
            save_count=10,  # Some caching thing
            blit=True,
            repeat=False
        )

        plt.show()
