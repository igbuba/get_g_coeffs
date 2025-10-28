import mcu
import numpy as np
from mcu import WAVECAR
#from mcu import vasprun
# This is for space group, specific band info, from pymatgen,  since i cant extract directly from wavecar with mcu, to be fixed later
from pymatgen.io.vasp import Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.electronic_structure.core import Spin

vasprun_dir    ="./"
vasprun_file   = vasprun_dir + "vasprun.xml"
l_wavecar_file = vasprun_dir + "WAVECAR"
output_file    = "g_coefs_cubic.txt"

# link to example: https://hungqpham.com/mcu/plottingwfn.html
#parameters-for-get-unk
# ---------------------- get gap, VBM, CBM, kpoints, cell_info, etc  etc
def get_general_info(vasprun_directory):
    """Return the band gap (in eV) from a VASP calculation in the specified directory.

    Args:
        vasp_directory (str): Path to the directory containing vasprun.xml.
    """
    vasp = mcu.VASP(vasprun_directory)
    #print(f"Elements         :   {str(vasp.element):<15}")
    #print(f"Number of atoms. :   {vasp.natom:<15}")
    #print(f"Number of bands  :   {vasp.nbands:<15}")
    #print(f"Cell vector a,b,c:   {str(vasp.cell[0]):<5}")
    #print(f"Cell vectors: a={vasp.cell[0][0]}  b={vasp.cell[0][1]}  c={vasp.cell[0][2]}")
    #print(f"SOC         :   {str(vasp.soc):<15}")
    #print(f"grids:   {vasp.ngrid}")

    return vasp.element, vasp.natom, vasp.soc, vasp.nelec, vasp.nbands, vasp.cell
#    return print(f"#Elements: {str(vasp.element):<15};  Number of atoms: {vasp.natom:<5};  Number of bands: {vasp.nbands:<5}")

Elements, n_atoms, is_soc, n_elec, n_bands, cell_vecs = get_general_info(vasprun_dir)
print(f"#Elements: {str(Elements):<15};   Number of bands: {n_bands:<5}; Number of atoms: {n_atoms:<5}; Cell vectors: a={cell_vecs[0][0]}  b={cell_vecs[0][1]}  c={cell_vecs[0][2]}\n")

# ---------- get space group from vasprun.xml ---------------
def get_space_group(file_name):
    sg = SpacegroupAnalyzer(Vasprun(file_name).final_structure)
    return sg.get_space_group_symbol(), sg.get_space_group_number()

symbol, number = get_space_group(vasprun_file) #"PBE/vasprun.xml")

# ------------ Get KS-vasprun.xml file.
def read_eigenvals_from_vasprun(file_path, n_cbm_plus=2, n_vbm_minus=2,
                                kpoints_list=None, verbose=False):
    """
    Reads eigenvalues around the VBM and CBM from a vasprun.xml file and returns
    band-edge and k-point resolved data.

    Returns
    -------
    tuple
        (gap, vbm_energy, cbm_energy,
         vbm_band_index, cbm_band_index,
         vbm_kpoint, cbm_kpoint,
         kpoints, band_indices, eigenvalues, occupancies)
    """
    vrun = Vasprun(file_path, parse_projected_eigen=False)
    bs = vrun.get_band_structure()

    # Band edges
    vbm = bs.get_vbm()
    cbm = bs.get_cbm()
    gap = bs.get_band_gap()

    vbm_band_index = vbm["band_index"][Spin.up][0]
    cbm_band_index = cbm["band_index"][Spin.up][0]
    vbm_kpoint = [float(round(x, 5)) for x in vbm["kpoint"].frac_coords]
    cbm_kpoint = [float(round(x, 5)) for x in cbm["kpoint"].frac_coords]

    if verbose:
        print(f"VBM: {vbm['energy']:.4f} eV at {vbm_kpoint}, band {vbm_band_index+1}")
        print(f"CBM: {cbm['energy']:.4f} eV at {cbm_kpoint}, band {cbm_band_index+1}")
        print(f"Band gap: {gap['energy']:.4f} eV\n")

    kpoints = vrun.actual_kpoints
    eigenvalues_data = vrun.eigenvalues[Spin.up]  # [nkpoints][nbands][2]

    # Select k-points
    if kpoints_list is not None:
        tol = 1e-3
        selected_indices = [
            i for i, kpt in enumerate(kpoints)
            if any(all(abs(kpt[j] - kp[j]) < tol for j in range(3)) for kp in kpoints_list)
        ]
    else:
        selected_indices = range(len(kpoints))

    kpt_list, bands_list, eigvals_list, occs_list = [], [], [], []

    for i in selected_indices:
        kpt = [float(round(x, 3)) for x in kpoints[i]]
        nband_total = len(eigenvalues_data[i])

        # Explicit band selection around VBM/CBM
        selected_bands = set()

        # Add VBM and its neighbors
        for b in range(vbm_band_index - n_vbm_minus, vbm_band_index + 1):
            if 0 <= b < nband_total:
                selected_bands.add(b)

        # Add CBM and its neighbors
        for b in range(cbm_band_index, cbm_band_index + n_cbm_plus + 1):
            if 0 <= b < nband_total:
                selected_bands.add(b)

        band_indices = sorted(selected_bands)

        eigvals = [float(eigenvalues_data[i][b][0]) for b in band_indices]
        occs = [float(eigenvalues_data[i][b][1]) for b in band_indices]
        bands = [b + 1 for b in band_indices]  # 1-based

        kpt_list.append(kpt)
        bands_list.append(bands)
        eigvals_list.append(eigvals)
        occs_list.append(occs)

    return (gap['energy'], vbm['energy'], cbm['energy'],
            vbm_band_index + 1, cbm_band_index + 1,
            vbm_kpoint, cbm_kpoint,
            kpt_list, bands_list, eigvals_list, occs_list)


(gap, vbm, cbm,
 vbm_band, cbm_band,
 vbm_kpt, cbm_kpt,
 kpts, bands, eigs, occs) = read_eigenvals_from_vasprun(
    vasprun_file,
    n_cbm_plus=0,
    n_vbm_minus=0,
    kpoints_list=[[0.0, 0.0, 0.0]],
    verbose=False
)

# ----- print result to test

print("\n--- Band Edge Info ---")
print(f"Band gap: {gap:.4f} eV")
print(f"VBM: {vbm:.4f} eV, band {vbm_band}, k-point {vbm_kpt}")
print(f"CBM: {cbm:.4f} eV, band {cbm_band}, k-point {cbm_kpt}")

print("\n--- Eigenvalues Around VBM/CBM ---")
for k, b, e, o in zip(kpts, bands, eigs, occs):
    print(f"k-point = {k}")
    print(f"band_indices = {b}")
    print(f"eigenvalues  = {e}")
    print(f"occupancies  = {o}\n")


list_n, eig_n, occu_n = bands[0], eigs[0], occs[0]
print("New II:", list_n, eig_n, occu_n )

#list_nn, eig_nn, occu_nn = bands[0], eigs[0], occs[0]
#print("New II:", list_nn, eig_nn, occu_nn )
#list_nn, eig_nn, occu_nn = zip( list(list_n), list(eig_n), list(occu_n))
# ------------- extract u_nk at k-points and for 6 nbands: cbm-3,2,1 vbm-1,2,3 --------

def extract_wavecar_data(wavecar_path, band_list, k_points):
    """
    Extract unk, gvec, and g_coeff for specific bands and k-point from a WAVECAR file.

    Args:
        wavecar_path (str): Path to the WAVECAR file.
        band_list (list): List of band indices (e.g., [17, 18]).
        k-points (int): Index of the k-point (e.g., 3 for R-point).

    Returns:
        unk (numpy.ndarray): Cell-periodic part of the wavefunctions.
        gvec (numpy.ndarray): G-vectors.
        g_coeff (numpy.ndarray): Plane wave coefficients.
    """
    mymcu = WAVECAR(wavecar_path)
    #unk  = mymcu.get_unk(band_list=[17,18], kpt=3)
    #print("unk shape",unk.shape)
    gvec = mymcu.get_gvec(kpt=k_points)
    g_coeff = mymcu.get_coeff(band_list=band_list, kpt=k_points)
    #print("g_vecs.shape. :", gvec.shape)
    #print("g_coeff.shape. :", g_coeff.shape)
    vasp = mcu.VASP(vasprun_dir)
    print("K-POINTS  :", vasp.kpts)
   # new_kp = mymcu.get_band.kpts 

    return gvec, g_coeff

# -------------- End of functions -------------------------------------- 

#list_eigen_nband = [[15, 16, 17, 18, 19, 20], 
#                   [-0.660241, -0.660241, 0.940883, 2.406590, 2.406590]]
input_text = """k-point   1 :   0.0000 0.0000 0.0000
band   eigenvalue(eV) occupancy
 66     -0.589743       1
 67     -0.138149       1
 68      0.441755       1
 69      2.552088       0
 70      2.561227       0
 71      2.588236       0"""
#list_n, eig_n, occu = zip(*[(int(line.split()[0]), float(line.split()[1]), int(line.split()[2])) for line in input_text.split('\n')[2:]])
#print(list(list_n), list(eig_n), list(occu))

for j, n in enumerate(list_n):

    # get Gvecs, g_coeff for cbm,vbm or band list [], and kpoints
    gvec, g_coeff = extract_wavecar_data(l_wavecar_file, band_list=[n], k_points=0)
    gvec=list(gvec)
    re_gcoeff = list(np.real(g_coeff))[0]
    im_gcoeff = list(np.imag(g_coeff))[0]
    # get band and general info here from vasprun.xml 
    Elements, n_atoms, is_soc, n_elec, n_bands, cell_vecs = get_general_info(vasprun_dir)
    # get space group and number vasprun.xml 
    symbol, number = get_space_group(vasprun_file) 

    if j==0:
        wmode='w'
    else:
        wmode='a'

    with open(output_file, wmode) as f:
        if j==0:
            
            f.write(f"# Space group      : {symbol} ({number})\n")
            #print(f"Space group: {symbol} ({number})")
            #f.write(f"# Atomic species   : {str(Elements):<5}\n# Number of atoms  : {n_atoms:<5}\n# Number of bands  : {n_bands:<5}\n# Cell vectors in Å: a={cell_vecs[0][0]},  b={cell_vecs[0][1]},  c={cell_vecs[0][2]}\n")
            f.write(
                    f"# Atomic species   : {str(Elements):<5}\n"
                    f"# Number of atoms  : {n_atoms:<5}\n"
                    f"# SOC              : {str(is_soc):<5}\n"
                    f"# Number of elec   : {n_elec:<5}\n"
                    f"# Number of bands  : {n_bands:<5}\n"
                    f"# Cell vectors in Å: a={cell_vecs[0][0]}, b={cell_vecs[0][1]}, c={cell_vecs[0][2]}\n"
                    f"# Band gap (eV)    : {gap:.4f} eV\n"
                    f"# VBM: {vbm:.4f} eV, band {vbm_band}, k-point {vbm_kpt}\n"
                    f"# CBM: {cbm:.4f} eV, band {cbm_band}, k-point {cbm_kpt}\n"
                    f" \n"

                    )
            f.write("#\tGx\tGy\tGz\tRe(C_nk)\tIm(C_nk)\n")

        # change range of g_coeffs by 1/2 if lsorbit = true  
        range_g_coeff = len(re_gcoeff)
        range_g_coeff = len(re_gcoeff) // (2 if is_soc else 1)
        if j == 0:
            print(f"Check len(re_gcoeff) : {range_g_coeff}, soc: {is_soc}")

        for i in range(range_g_coeff):
        #for i in range(len(re_gcoeff)):
            if i==0:
                text=f'n_band : {n},\tenergy : {eig_n[j]},\toccupation : {occu_n[j]}\n'
            else:
                text=''
            f.write(f"{text}\t{gvec[i][0]}\t{gvec[i][1]}\t{gvec[i][2]}\t{round(float(re_gcoeff[i]), 8):13.8f}     {round(float(im_gcoeff[i]), 8):13.8f}\n")

        f.write('\n')
