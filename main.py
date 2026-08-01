from toptimizer import beso, top

import numpy as np

if __name__ == "__main__":
    # 1: MBB half beam
    # 2: 2-force cantilever
    # 3: Cantilever with hole
    case = 3

    if case == 1:
        design = top.Design(60, 20)

        # Downwards force on the top-left corner
        design.add_forces([0], [0], forces_y=[-1])

        # Fix the bottom-right corner in the y direction
        design.add_fixed(top.Fix.Y, [design.nelx], [design.nely])
        # Fix the left wall in the x direction
        design.add_fixed(top.Fix.X,
            [0] * (design.nely + 1),
            range(design.nely + 1),
        )

        optimizer = beso.beso(design)
    elif case == 2:
        design = top.Design(30, 30)

        # Fix the left wall in x and y
        design.add_fixed(top.Fix.XY,
            [0] * (design.nely + 1),
            range(design.nely + 1)
        )

        # Upwards force on top-left and downwards force on bottom-right
        design.add_forces(
            [design.nelx, design.nelx],
            [0, design.nely],
            forces_y=[1, -1]
        )

        params = beso.BesoParams(volfrac=0.4, rmin=1.2)

        optimizer = beso.beso(design, params)
    elif case == 3:
        design = top.Design(45, 30)

        # Fix the left wall in x and y
        design.add_fixed(top.Fix.XY,
            [0] * (design.nely + 1),
            range(design.nely + 1)
        )

        # Force a circular hole in the design
        for ely in range(design.nely):
            for elx in range(design.nelx):
                d = ((ely - design.nely / 2)**2 + (elx - design.nelx / 3)**2)**0.5
                if d < design.nely / 3:
                    design.add_passive(top.Passive.VOID, [elx], [ely])

        # Downwards force on bottom-right
        design.add_forces(
            [design.nelx],
            [design.nely],
            forces_y=[-1]
        )

        params = beso.BesoParams(volfrac=0.5, ert=0.04)

        optimizer = beso.beso(design, params)
    else:
        print(f"Error: invalid case selected for boundary conditions: {case}")
        exit(1)

    top.animate(design, optimizer)
