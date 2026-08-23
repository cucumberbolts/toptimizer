# toptimizer
Topology optimization code written in Python. Includes implementations of OC (Optimality Criteria) and BESO (Bi-directional Evolutionary Structural Optimization).

## Running
Clone the repository: 
```bash
git clone https://github.com/cucumberbolts/toptimizer
cd toptimizer
```

Create a virtual environment and install dependencies:
```bash
python -m venv venv

source venv/bin/activate # Unix
venv/Scripts/activate.bat # Windows

pip install -r requirements.txt
```

The file `main.py` contains a quick demo on how to use the BESO topology optimizer. There are 3 predefined boundary conditions which can be changed by setting the `case` variable. You may also define your own. The choice between BESO and OC to update the design variables can be chosen with the `algorithm` variable.
```bash
python main.py
```

![OC 2 Force Cantilever](https://github.com/cucumberbolts/toptimizer/blob/assets/OC_2_Force_Cantilever.gif)

## Usage
Toptimizer comes with a basic Python API to define loading conditions and
optimize using BESO and OC, found in the `beso` and `oc` modules respectively.
General functionality common to both optimizers is found in the `top` module.

### Boundary Conditions
The design domain and its boundary conditions are stored in the `top.Design`
class where constraints can be added on in a step-by-step manner. The design
is a rectangular grid consisting of square elements.
  
Nodes are defined as the corners of the elements, so a design with `nelx`
elements across and `nely` elements down would have `nelx + 1` nodes across
and `nely + 1` elements down.

```
Example where nelx = 4 and nely = 3

Elements:
     0   1   2   3
   +---+---+---+---+
 0 |   |   |   |   |
   +---+---+---+---+
 1 |   |   |   |   |
   +---+---+---+---+
 2 |   |   |   |   |
   +---+---+---+---+

Nodes:
   0   1   2   3   4
 0 +---+---+---+---+
   |   |   |   |   |
 1 +---+---+---+---+
   |   |   |   |   |
 2 +---+---+---+---+
   |   |   |   |   |
 3 +---+---+---+---+
```
  
### Short Example: MBB Half Beam with BESO
```python
from toptimizer import top, beso

# Create a design domain 60 elements wide and 20 elements tall
NELX, NELY = 60, 20
design = top.Design(NELX, NELY)

# Apply a downwards force on the top-left corner
# (This node's coordinate is x=0, y=0)
design.add_forces(
    [0],  # x
    [0],  # y
    forces_y=[-1]  # Downwards force of magnitude 1
    # add_forces() will automatically fill in forces_x with zeros if not given
)

# Fix the bottom-right corner in the y direction
design.add_fixed(top.Fix.Y,
    [design.nelx], # x coordinate
    [design.nely], # y coordinate
)
# Fix the left wall in the x direction
design.add_fixed(top.Fix.X,
    [0] * (design.nely + 1),  # x coordinates
    range(design.nely + 1),   # y coordinates
)

optimizer = beso.Beso(design, volfrac=0.3, rmin=2.5)
top.animate(optimizer)
```

![BESO MBB Half Beam](https://github.com/cucumberbolts/toptimizer/blob/assets/BESO_MBB_Half_Beam.gif)
  
A more extensive exposition of the available functionality can be seen in the `main.py` file.

## References
Several papers were referenced, found in the `papers/` folder. The code from these papers can be found in `ext/`.

- Andreassen, Erik, et al. “Efficient topology optimization in MATLAB USING 88 lines of code.” Structural and Multidisciplinary Optimization, vol. 43, no. 1, 20 Nov. 2010, pp. 1–16, https://doi.org/10.1007/s00158-010-0594-7. 
- Sigmund, O. “A 99 line topology optimization code written in MATLAB.” Structural and Multidisciplinary Optimization, vol. 21, no. 2, Apr. 2001, pp. 120–127, https://doi.org/10.1007/s001580050176. 
- Zuo, Zhi Hao, and Yi Min Xie. “A simple and compact Python code for Complex 3D topology optimization.” Advances in Engineering Software, vol. 85, July 2015, pp. 1–11, https://doi.org/10.1016/j.advengsoft.2015.02.006. 

Additionally, some existing implementations were also referenced:
- https://github.com/ToddyXuTao/BESO-for-2D/
