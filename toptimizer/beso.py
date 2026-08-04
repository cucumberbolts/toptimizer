"""
Implementation of BESO optimization algorithm

July 2026
Julian Poon
"""

from dataclasses import dataclass

from typing import Generator

from .top import Design, Passive

import numpy as np
from scipy.sparse import csc_array, csr_array
from scipy.sparse.linalg import spsolve


@dataclass(kw_only=True, frozen=True)
class BesoParams:
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


def fea_loop(x: np.array, design: Design, penal: float) -> np.array:
    """
    Perform finite element analysis
    to obtain the displacement vector

    Parameters
    ----------
    x : np.array
        2D material density distribution for each element
    design : Design
        Boundary conditions for the design
    penal : float
        Penalization factor (usually 3)

    Returns
    -------
    np.array
        The global displacement vector
    """

    nely, nelx = x.shape

    dof_count = 2 * (nelx + 1) * (nely + 1)

    # Force matrix, where each column is a force vector
    F = design.get_forces()
    # Global stiffness matrix
    # K = dok_array((dof_count, dof_count))
    K = np.zeros((dof_count, dof_count))
    # Global displacement vector
    U = np.zeros((dof_count, F.shape[1]))

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

    y = spsolve(
        K[np.ix_(free_dofs, free_dofs)],
        F[np.array(free_dofs), :]
    )

    # TODO: There HAS to be a better way...
    if isinstance(y, np.ndarray):
        U[np.array(free_dofs), :] = y.reshape((-1, 1))
    else:
        U[np.array(free_dofs), :] = y.todense()

    U[np.ix_(list(design.fixed))] = 0

    return U


k_row, k_col, elem_dofs = None, None, None

def fea_init(nelx: int, nely: int):
    """
    Prepare FEA
    TODO: refactor
    """
    global k_row, k_col, elem_dofs

    # DEGREES OF FREEDOM
    # ------------------
    # Each element has 8 degrees of freedom (DOFs)
    # --one for each x and y component of each node
    #
    # An example numbering of degrees of freedom for
    # a 1x2 design domain (nelx = 2, nely = 1):
    # (0, 1) (2, 3) (4, 5)
    #    +------+------+
    #    |      |      |
    #    |  e1  |  e2  |
    #    +------+------+
    # (6, 7) (8, 9) (10, 11)
    # 
    # DOFs are indexed in a clockwise fashion, starting
    # with the top-left node's x-component. For example,
    # the DOFs of e1 are: [0, 1, 2, 3, 8, 9, 6, 7]

    # The first degree of freedom number for each element (x-component of the top-left node)
    first_dofs = 2 * np.array([y * (nelx + 1) + x for y in range(nely) for x in range(nelx)])
    # DOFs for element 0, can be added to the first DOF of
    # each element in 'first_dofs' to give that element's DOFs
    template_dof = np.array([0, 1, 2, 3, 2, 3, 0, 1]); template_dof[4:] += 2 * (nelx + 1)
    # Matrix whose row entries give the element's degrees of freedom (row number = element number)
    elem_dofs = np.broadcast_to(first_dofs[:, np.newaxis] + template_dof[np.newaxis, :], (nelx * nely, 8))

    # Indices into the global stiffness matrix K
    # such that K[k_col[i], k_row[i]] = k_data[i]
    k_row = np.repeat(elem_dofs, 8)
    k_col = np.tile(elem_dofs, 8).reshape(-1)

def fea(x: np.array, design: Design, penal: float) -> np.array:
    """
    Perform finite element analysis to
    obtain the global displacement vector.

    Parameters
    ----------
    x : np.array
        2D material density distribution for each element
    design : Design
        Boundary conditions for the design
    penal : float
        Penalization factor (usually 3)

    Returns
    -------
    np.array
        The global displacement vector
    """

    nely, nelx = x.shape

    dof_count = 2 * (nelx + 1) * (nely + 1)

    # Force matrix, where each column is a force vector
    F = design.get_forces()
    # Global displacement vector
    force_count = F.shape[1]
    U = np.zeros((dof_count, force_count))

    # Entries in the global stiffness matrix K
    k_data = np.reshape(np.outer(x**penal, K_LOCAL), -1)

    # Note: indices appearing multiple times are summed
    K = csc_array((k_data, (k_row, k_col)))
    # K = (K + K.T) * 0.5

    free_dofs = [d for d in range(dof_count) if d not in design.fixed]

    y = spsolve(
        K[np.ix_(free_dofs, free_dofs)],
        F[np.array(free_dofs), :]
    )

    # TODO: There HAS to be a better way...
    if isinstance(y, np.ndarray):
        U[np.array(free_dofs), :] = y.reshape((-1, 1))
    else:
        U[np.array(free_dofs), :] = y.todense()

    U[np.ix_(list(design.fixed))] = 0

    return U


def sensitivity_loop(x: np.array, U: np.array, params: BesoParams) -> tuple[np.array, float]:
    """
    Calculate the sensitivities of each element
    c = Ui^T * Ki * Ui

    Parameters
    ----------
    x : np.array
        Array of design variables, 'SOLID' for solid, 'VOID' for void
    U : np.array
        Global displacement vector

    Returns
    -------
    tuple[np.array, float]
        Array of sensitivity values for each element in the design domain
        and the total compliance under the load
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

            for i in range(U.shape[1]):
                # Get the element displacement from the global displacement vector
                ue = U[np.array(edof), i]

                # Calculate the compliance
                # c = 0.5 * ue.T @ K_LOCAL @ ue
                ce = ue.T @ K_LOCAL @ ue
                c += x[ely, elx] ** params.penal * ce

                # Calculate the sensitivity
                # dc[ely, elx] += -params.penal * x[ely, elx] ** (params.penal - 1) * ce
                dc[ely, elx] += x[ely, elx] ** (params.penal - 1) * ce

    # Filter the sensitivities
    dcf = filter_loop(x, params.rmin, dc)

    return dcf, c

def sensitivity(x: np.array, U: np.array, params: BesoParams) -> tuple[np.array, float]:
    """
    Calculate the sensitivities of each element
    c = Ui^T * Ki * Ui

    Parameters
    ----------
    x : np.array
        Array of design variables, 'SOLID' for solid, 'VOID' for void
    U : np.array
        Global displacement vector

    Returns
    -------
    tuple[np.array, float]
        Array of sensitivity values for each element in the design domain
        and the total compliance under the load
    """

    nely, nelx = x.shape

    c = 0.0
    dc = np.zeros(x.shape)

    # Loop over displacements from all forces
    for i in range(U.shape[1]):
        # Compliance of each element
        # Computes ce = ue^T * ke * ue for each element
        ce = np.sum(U[elem_dofs, i] @ K_LOCAL * U[elem_dofs, i], axis=1).reshape(x.shape)
        # Global compliance
        # The global compliance (objective function) C = sum(x^penal * ce)
        c += np.sum(x**params.penal * ce)
        # Sensitivity of each element
        dc += x ** (params.penal - 1) * ce

    # Filter the sensitivities
    dcf = np.reshape(H @ np.reshape(dc, -1), dc.shape)

    return dcf, c

def filter_loop(x: np.array, rmin: float, dc: np.array) -> np.array:
    """
    Filters the sensitivities

    Parameters
    ----------
    x : np.array
        Array of design variables, 'SOLID' for solid, 'VOID' for void
    rmin : float
        Filter radius
    dc : np.array
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

H = None

def filter_init(nelx: int, nely: int, rmin: float):
    """
    Prepare the sensitivity filter
    TODO: refactor
    """

    f = int(np.floor(rmin))

    # A slightly large bounding box of the elements that are considered by the filter
    square_size = int((2 * np.ceil(rmin)) ** 2)
    size = nelx * nely * square_size
    h_row = np.zeros(size)
    h_col = np.zeros(size)
    h_data = np.zeros(size)

    idx = 0

    for i in range(nely):
        for j in range(nelx):
            centre_elem = i * nelx + j
            for k in range(max(i - f, 0), min(i + f + 1, nely)):
                for l in range(max(j - f, 0), min(j + f + 1, nelx)):
                    elem = k * nelx + l
                    fac = max(0, rmin - np.sqrt((i - k)**2 + (j - l)**2))

                    h_row[idx] = centre_elem
                    h_col[idx] = elem
                    h_data[idx] = fac

                    idx += 1

    global H
    H = csr_array((h_data[:idx], (h_row[:idx], h_col[:idx])), shape=(nelx * nely, nelx * nely))

    # Divide each row by its sum
    H /= H.sum(axis=1)[:, np.newaxis]


def update(x: np.array, dc: np.array, vol: float, passive: np.array) -> np.array:
    """
    Updates the design variables given filtered sensitivities
    and target volume fraction using the bisection algorithm.

    Parameters
    ----------
    x : np.array
        Array of design variables, 'SOLID' for solid, 'VOID' for void
    dc : np.array
        Array of filtered sensitivities
    vol : float
        Target volume
    passive : np.array
        Array defining passive elements in x

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
        x = np.maximum(np.tile(VOID, x.shape), np.sign(dc - threshold))

        # for ely in range(nely):
        #     for elx in range(nelx):
        #         if dc[ely, elx] > threshold:
        #             x[ely, elx] = SOLID
        #         else:
        #             x[ely, elx] = VOID

        # Passive elements
        x[passive==Passive.VOID] = VOID
        x[passive==Passive.SOLID] = SOLID

        if x.sum() > vol:
            low = threshold
        else:
            high = threshold

    return x


def beso(design: Design, params: BesoParams = None) -> Generator[np.array, None, None]:
    """
    Optimize topology using the BESO algorithm and
    yield the design variable values for each iteration

    Parameters:
    -----------
    design : Design
        The boundary conditions of the design
    params : BesoParams
        Parameters for the BESO algorithm

    Returns:
    --------
    Generator[np.array, None, None]
        A generator object that yields each
        iteration of the design variables
    """

    if params is None:
        params = BesoParams()

    # Initialize iteration
    x  = np.tile(SOLID, (design.nely, design.nelx))  # Design variables
    dc = np.zeros((design.nely, design.nelx))        # Sensitivity values

    fea_init(design.nelx, design.nely)
    filter_init(design.nelx, design.nely, params.rmin)

    c_hist = []  # Keep track of compliance values

    current_vol = design.nelx * design.nely * SOLID  # Target volume for current iteration
    target_vol  = design.nelx * design.nely * params.volfrac  # Target volume of final design

    # Force passive void elements to be void
    # (Everything is already initialized to solid)
    x[design.passive==Passive.VOID] = VOID

    it = 0
    change = 1

    # Start Iteration
    while change > 0.001:
        # Finite element analysis
        U = fea(x, design, params.penal)

        # Sensitivity analysis
        dc_old = np.copy(dc)
        dc, c = sensitivity(x, U, params)

        c_hist.append(c)

        # Average the sensitivities with the previous iteration
        if it > 0:
            dc = 0.5 * (dc + dc_old)

        # Update the current iteration's target volume
        # according to the evolutionary rate
        if current_vol > target_vol:
            current_vol *= 1 - params.ert

        # Update the design variables according to sensitivity analysis
        x = update(x, dc, current_vol, design.passive)

        # Check for convergence and log information
        vol = x.sum() / design.nelx / design.nely
        if it > 9:
            old_c = sum(c_hist[it-9:it-4])
            new_c = sum(c_hist[it-4:])
            change = np.abs((new_c - old_c) / old_c)

            print(f"Iteration: {it} Compliance: {c:10.4f} Volume: {vol:6.3f} Change: {change:6.3f}")
        else:
            print(f"Iteration: {it} Compliance: {c:10.4f} Volume: {vol:6.3f}")

        it += 1

        yield x
