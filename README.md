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

## References
Several papers were referenced, found in the `papers/` folder. The code from these papers can be found in `ext/`.

- Andreassen, Erik, et al. “Efficient topology optimization in MATLAB USING 88 lines of code.” Structural and Multidisciplinary Optimization, vol. 43, no. 1, 20 Nov. 2010, pp. 1–16, https://doi.org/10.1007/s00158-010-0594-7. 
- Sigmund, O. “A 99 line topology optimization code written in MATLAB.” Structural and Multidisciplinary Optimization, vol. 21, no. 2, Apr. 2001, pp. 120–127, https://doi.org/10.1007/s001580050176. 
- Zuo, Zhi Hao, and Yi Min Xie. “A simple and compact Python code for Complex 3D topology optimization.” Advances in Engineering Software, vol. 85, July 2015, pp. 1–11, https://doi.org/10.1016/j.advengsoft.2015.02.006. 

Additionally, some existing implementations were also referenced:
- https://github.com/ToddyXuTao/BESO-for-2D/
