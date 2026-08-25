"""
Informal test cases and demo for the beso module
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

from pyinstrument import Profiler

from toptimizer import beso, oc, top

if __name__ == "__main__":
    # 1: MBB half beam
    # 2: 2-force cantilever
    # 3: Cantilever with hole
    case = 2

    # 1: BESO
    # 2: OC
    algorithm = 2

    profile = False

    save_file = False

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

        if algorithm == 1:
            optimizer = beso.Beso(design, rmin=2.5, volfrac=0.3)
            file_name = "BESO_MBB_Half_Beam.gif"
        elif algorithm == 2:
            optimizer = oc.Oc(design, volfrac=0.3)
            file_name = "OC_MBB_Half_Beam.gif"
    elif case == 2:
        design = top.Design(60, 60)

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

        if algorithm == 1:
            optimizer = beso.Beso(design, volfrac=0.4, rmin=1.2)
            file_name = "BESO_2_Force_Cantilever.gif"
        elif algorithm == 2:
            optimizer = oc.Oc(design, volfrac=0.4, rmin=1.2)
            file_name = "OC_2_Force_Cantilever.gif"
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

        optimizer = beso.Beso(design, volfrac=0.5, ert=0.04)
        file_name = "BESO_Cantilever_With_Hole.gif"
    else:
        print(f"Error: invalid case selected for boundary conditions: {case}")
        sys.exit(1)

    if profile:
        fig, ax = plt.subplots()
        im = ax.matshow(np.zeros((design.nely, design.nelx)))

        # Specify the upper and lower bounds of the values to
        # be plotted so the colours can be displayed properly
        im.set_clim(0.0, 1.0)

        with Profiler(interval=0.05) as profiler:
            for x in optimizer:
                final_image = x

        profiler.print()

        ax.matshow(final_image)

        plt.show()
    else:
        if save_file:
            top.animate(optimizer, file_name, fps=12)
        else:
            top.animate(optimizer)
