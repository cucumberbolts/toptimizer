from toptimizer import beso, top

import numpy as np

if __name__ == "__main__":
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

    top.animate(design, optimizer)
