"""
Implementation of OC optimization algorithm
"""

from typing import Generator

import numpy as np
from scipy.sparse import csc_array, csr_array
from scipy.sparse.linalg import spsolve

from .top import Design, Passive


class Oc:
    """
    Iterable object that successively yields the next
    iteration of the optimization loop calculated using
    the OC (Optimality Criteria) algorithm
    """

    def __init__(self, design: Design, *,
            penal: float   = 3.0,
            volfrac: float = 0.3,
            rmin: float    = 1.5
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
        """

        self.penal = penal
        self.volfrac = volfrac
        self.rmin = rmin

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
        square_size = (2 * int(np.ceil(rmin) + 1)) ** 2
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
        Optimize topology using the OC algorithm and
        yield the design variable values for each iteration

        Returns:
        --------
        Generator[np.array, None, None]
            A generator object that yields each
            iteration of the design variables
        """

        self.x = np.tile(self.volfrac, (self.design.nely, self.design.nelx))  # Design variables

        nel = self.design.nelx * self.design.nely

        it = 0
        change = 1

        while change > 0.01:
            x_old = np.copy(self.x)

            U = self.fea()

            # Perform sensitivity analysis
            dc, c = self.sensitivity(U)

            # Filter the sensitivity values
            dc = self.check(dc)

            # Design update by the optimality criteria method
            self.x = self.update(dc)

            change = abs(self.x - x_old).max()

            print(f"Iteration: {it} Compliance: {c:10.4f} Volume: {self.x.sum() / nel:6.3f} Change: {change:6.3f}")

            it += 1

            yield np.copy(self.x)

        print("Done!")


    def fea(self) -> np.array:
        """
        Perform finite element analysis
        to obtain displacement vector

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


    def sensitivity(self, U: np.array) -> np.array:
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
            # ce = ue^T * ke * ue
            ce = 0.5 * np.sum(U[self.elem_dofs, i] @ self.ke * U[self.elem_dofs, i], axis=1).reshape(self.x.shape)

            # Compute the global compliance, c
            # c = sum(x^penal * ce)
            c += np.sum(self.x**self.penal * ce)

            # Compute the sensitivity of each element, dc
            # dc = -penal * xe^(penal - 1) * ue^T * ke * ue
            #    = -penal * xe^(penal - 1) * ce
            dc += -self.penal * self.x ** (self.penal - 1) * ce

        return dc, c


    def check(self, dc: np.array) -> np.array:
        """
        Filter sensitivities by applying the mesh-independency filter.
        This is done to avoid the "checkerboarding problem"

        Parameters:
        -----------
        dc : np.array
            Sensitivity of the compliance (objective function) with respect to the design variables x

        Returns:
        --------
        np.array:
            The filtered sensitivities. These are modified
            versions of the original sensitivities dc after
            appllying the mesh-independency filter.
        """

        dcf = np.reshape(self.filter @ np.reshape(dc * self.x, -1), dc.shape) / self.x
        return dcf


    def update(self, dc: np.array) -> np.array:
        """
        Updates the density values of the design
        variables with optimality criteria method

        Parameters
        ----------
        dc : np.array
            Array of filtered sensitivities

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

        x_new = np.zeros(self.x.shape)

        # Bisection method to find Lagrange multiplier
        while l_high - l_low > 0.0001:
            l_mid = 0.5 * (l_low + l_high)

            x_new = np.maximum(0.001, np.maximum(self.x - move, np.minimum(1, np.minimum(self.x + move, self.x * np.sqrt(-dc / l_mid)))))

            if np.sum(x_new) - self.volfrac * self.design.nelx * self.design.nely > 0:
                l_low = l_mid
            else:
                l_high = l_mid

        return x_new
