"""
Implementation of BESO optimization algorithm
"""

from typing import Generator

import numpy as np
from scipy.sparse import csc_array, csr_array
from scipy.sparse.linalg import spsolve

from .top import Design, Passive, IterInfo


SOLID = 1.0
VOID = 0.001


class Beso:
    """
    Iterable object that successively yields the next
    iteration of the optimization loop calculated using the BESO
    (Bi-directional Evolutionary Structural Optimization) algorithm
    """

    def __init__(self, design: Design, *,
            penal: float   = 3.0,
            volfrac: float = 0.3,
            rmin: float    = 1.5,
            ert: float     = 0.02
        ):

        """
        Initialize design parameters and boundary conditions as
        well as data later used for FEA and sensitivity analysis

        Parameters:
        -----------
        design : Design
            The boundary conditions of the design
        penal : float
            Penalization exponent for FEA
        volfrac : float
            Volume fraction of the optimized design
        rmin : float
            Sensitivity filter radius
        ert : float
            Evolutionary rate of BESO
        """

        self.penal = penal
        self.volfrac = volfrac
        self.rmin = rmin
        self.ert = ert

        self.design = design

        #######################################
        ### Assemble local stiffness matrix ###
        #######################################

        # Coefficients for the local stiffness matrix
        k = np.array([
            1/2-design.nu/6,   1/8+design.nu/8, -1/4-design.nu/12, -1/8+3*design.nu/8,
           -1/4+design.nu/12, -1/8-design.nu/8,       design.nu/6,  1/8-3*design.nu/8,
        ])

        # Local stiffness matrix for each element
        self.ke = design.E/(1 - design.nu**2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        ])

        ###################
        ### Prepare FEA ###
        ###################

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

        nely, nelx = self.design.nely, self.design.nelx

        # The first degree of freedom number for each element (x-component of the top-left node)
        first_dofs = 2 * np.array([y * (nelx + 1) + x for y in range(nely) for x in range(nelx)])
        # Offset values from an element's first DOF
        template_dof = np.array([0, 1, 2, 3, 2, 3, 0, 1]); template_dof[4:] += 2 * (nelx + 1)

        # Matrix whose row entries give that element's degrees of freedom (row number = element number)
        self.elem_dofs = np.broadcast_to(first_dofs[:, np.newaxis] + template_dof[np.newaxis, :], (nelx * nely, 8))

        # Indices into the global stiffness matrix K
        # such that K[k_col[i], k_row[i]] = k_data[i]
        self.k_row = np.repeat(self.elem_dofs, 8)
        self.k_col = np.tile(self.elem_dofs, 8).reshape(-1)

        ######################
        ### Prepare filter ###
        ######################

        f = int(np.floor(rmin))

        # A slightly larger-than-needed bounding box of
        # the elements that are considered by the filter
        square_size = (2 * int(np.ceil(rmin + 1))) ** 2
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

        self.filter = csr_array((h_data[:idx], (h_row[:idx], h_col[:idx])), shape=(nelx * nely, nelx * nely))

        # Divide each row by that row's sum
        self.filter /= self.filter.sum(axis=1)[:, np.newaxis]


    def __iter__(self) -> Generator[np.array, None, None]:
        """
        Optimize topology using the BESO algorithm and
        yield the design variable values for each iteration

        Returns:
        --------
        Generator[np.array, None, None]
            A generator object that yields each
            iteration of the design variables
        """

        self.x = np.tile(SOLID, (self.design.nely, self.design.nelx))  # Design variables

        # Force passive void elements to be void
        # (Everything is already initialized to solid)
        self.x[self.design.passive==Passive.VOID] = VOID

        # Keep track of compliance values
        self.c_hist = []

        dc = np.zeros(self.x.shape)  # Sensitivity values

        nel = self.design.nelx * self.design.nely

        current_vol = nel * SOLID  # Target volume for current iteration
        target_vol  = nel * self.volfrac  # Target volume of final design

        it = 0
        change = 1

        # Start Iteration
        while change > 0.001:
            # Finite element analysis
            U = self.fea()

            # Sensitivity analysis
            dc_old = np.copy(dc)
            dc, c = self.sensitivity(U)

            self.c_hist.append(c)

            # Average the sensitivities with the previous iteration
            if it > 0:
                dc = 0.5 * (dc + dc_old)

            # Update the current iteration's target volume
            # according to the evolutionary rate
            if current_vol > target_vol:
                current_vol *= 1.0 - self.ert

            # Update the design variables according to sensitivity analysis
            self.update(dc, current_vol)

            # Check for convergence and log information
            vol = self.x.sum() / nel
            if it > 9:
                old_c = sum(self.c_hist[it-9:it-4])
                new_c = sum(self.c_hist[it-4:])
                change = np.abs((new_c - old_c) / old_c)

                print(f"Iteration: {it} Compliance: {c:10.4f} Volume: {vol:6.3f} Change: {change:6.3f}")
            else:
                print(f"Iteration: {it} Compliance: {c:10.4f} Volume: {vol:6.3f}")

            iter = it
            it += 1

            yield IterInfo(np.copy(self.x), iter, c, self.x.sum() / nel, change)

        print("Done!")


    def fea(self) -> np.array:
        """
        Perform finite element analysis to
        obtain the global displacement vector.

        Returns
        -------
        np.array
            The global displacement vector
        """

        nely, nelx = self.x.shape

        dof_count = 2 * (nelx + 1) * (nely + 1)

        # Force matrix, where each column is a force vector
        F = self.design.forces
        # Global displacement vector
        force_count = F.shape[1]
        U = np.zeros((dof_count, force_count))

        # Entries in the global stiffness matrix K
        k_data = np.reshape(np.outer(self.x**self.penal, self.ke), -1)

        # Note: indices appearing multiple times are summed
        K = csc_array((k_data, (self.k_row, self.k_col)))
        # K = (K + K.T) * 0.5

        free_dofs = [d for d in range(dof_count) if d not in self.design.fixed]

        U_temp = spsolve(
            K[np.ix_(free_dofs, free_dofs)],
            F[np.array(free_dofs), :]
        )

        # TODO: There HAS to be a better way...
        if isinstance(U_temp, np.ndarray):
            U[np.array(free_dofs), :] = U_temp.reshape((-1, 1))
        else:
            U[np.array(free_dofs), :] = U_temp.todense()

        U[np.ix_(list(self.design.fixed))] = 0

        return U


    def sensitivity(self, U: np.array) -> tuple[np.array, float]:
        """
        Calculate the sensitivity of each element

        Parameters
        ----------
        U : np.array
            Global displacement vector

        Returns
        -------
        tuple[np.array, float]:
            Array of sensitivity values for each element in the design domain
            and the total compliance under the load
        """

        c = 0.0
        dc = np.zeros(self.x.shape)

        # Loop over displacements caused by all forces
        for i in range(U.shape[1]):
            # Note: the 'e' subscript denotes an element-wise (local) calculation

            # For each element, compute the compliance, ce
            # ce = 0.5 * ue^T * ke * ue
            ce = 0.5 * np.sum(U[self.elem_dofs, i] @ self.ke * U[self.elem_dofs, i], axis=1).reshape(self.x.shape)

            # Compute the global compliance, c
            # c = sum(x^penal * ce)
            c += np.sum(self.x**self.penal * ce)

            # Compute the sensitivity of each element, dc
            # dc = -penal * xe^(penal - 1) * 0.5 * ue^T * ke * ue
            #    = -penal * xe^(penal - 1) * ce
            # Note that BESO only cares about the relative ranking
            # of the elements, so the -penal factor can be dropped
            # to make the sensitivity values positive (easier to rank)
            dc += self.x ** (self.penal - 1) * ce

        # Filter the sensitivities
        dcf = np.reshape(self.filter @ np.reshape(dc, -1), dc.shape)

        return dcf, c


    def update(self, dc: np.array, vol: float):
        """
        Updates the design variables given filtered sensitivities
        and target volume using the bisection algorithm

        Parameters
        ----------
        dc : np.array
            Array of filtered sensitivities
        vol : float
            Target volume
        """

        # Sensitivity threshold:
        # Variables with sensitivities lower than the threshold are made void and
        # variables with sensitivities higher than the threshold are made solid.
        # A higher sensitivity value indicates a greater decrease in compliance
        # when the design variable is increased. Note this is because the -penal
        # factor is dropped in the sensitivity analysis.
        threshold = 0.0

        # Find the sensitivity threshold with the bisection algorithm
        low  = dc.min()
        high = dc.max()

        while (high - low) / high > 0.00001:
            threshold = (low + high) * 0.5

            self.x[:] = VOID
            self.x[dc > threshold] = SOLID

            # Passive elements
            self.x[self.design.passive==Passive.VOID] = VOID
            self.x[self.design.passive==Passive.SOLID] = SOLID

            if self.x.sum() > vol:
                low = threshold
            else:
                high = threshold
