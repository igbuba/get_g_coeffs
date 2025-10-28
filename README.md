# get_g_coeff.py
A **Python script** to extract from VASP **WAVECAR** and **vasprun.xml** files:

- **G**-vectors
- Complex coefficients **C<sub>n,k</sub>(**G**)**

for any **k-point** (e.g., Γ) or set of k-points, and **band** (VBM, CBM) or any set of bands.

Useful for constructing basis sets for **k·p models**.

---

### **Code Sources**
- **WAVECAR extraction**: Adapted from [Hung Pham's MCU code](https://github.com/hungpham2017/mcu).
- **vasprun.xml extraction**: Uses [pymatgen](https://github.com/materialsproject/pymatgen).
