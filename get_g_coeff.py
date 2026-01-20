#from mcu-->  https://hungqpham.com/mcu/plottingwfn.html
import mcu
import numpy as np
from mcu import WAVECAR
from pymatgen.io.vasp import Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.electronic_structure.core import Spin

#vasprun_dir    ="/Users/ibuba/Desktop/dielectric_hybrid/Project_II/0Simon/orthorh/PBE/no_SOC/"
#vasprun_dir    ="/Users/ibuba/Desktop/dielectric_hybrid/Project_II/0Simon/cubic_/PBE_poly/no_SOC/"
vasprun_dir    ="/Users/ibuba/Desktop/dielectric_hybrid/Project_II/0Simon/Materials/cubic/PBE/"
vasprun_file   = vasprun_dir + "vasprun.xml"
l_wavecar_file = vasprun_dir + "WAVECAR"
output_file    = "PBE_g_coefs_cubic_10_BANDS.txt"

# ---------------------- get gap, VBM, CBM, kpoints, cell_info, etc  etc
def get_general_info(vasprun_directory):
    """Return the band gap, cell_info, etc from a VASP calculation in the specified directory.

    Args:
        vasp_directory (str): Path to the directory containing vasprun.xml.
    """
    vasp = mcu.VASP(vasprun_directory)
    #print(f"Elements         :   {str(vasp.element):<15}")
    #print(f"Number of atoms. :   {vasp.natom:<15}")

    return vasp.element, vasp.natom, vasp.soc, vasp.nelec, vasp.nbands, vasp.cell

Elements, n_atoms, is_soc, n_elec, n_bands, cell_vecs = get_general_info(vasprun_dir)
print(f"#Elements: {str(Elements):<15};   Number of bands: {n_bands:<5}; Number of atoms: {n_atoms:<5}; Cell vectors: a={cell_vecs[0][0]}  b={cell_vecs[0][1]}  c={cell_vecs[0][2]}\n")

# ---------- get space group and type of functional from vasprun.xml ---------------
def get_space_group_and_functional(file_name):
    """
    Returns space group information and functional type from a vasprun.xml file.

     """
    vrun = Vasprun(file_name)
    sg = SpacegroupAnalyzer(vrun.final_structure)
    incar = vrun.incar
    parameters = vrun.parameters or {}

    # Get functional parameters safely
    aexx = float(parameters.get("AEXX", incar.get("AEXX", 0.0)))
    hf_screen = float(parameters.get("HFSCREEN", incar.get("HFSCREEN", 0.0)))
    lhfcalc = incar.get("LHFCALC", False)

    run_type = "Unknown"

    # --- Identify type of functional ---
    if abs(aexx - 1.00) < 1e-3:
        run_type = "HF"
    elif abs(hf_screen - 0.30) < 1e-3:
        run_type = "HSE03"
    elif abs(hf_screen - 0.20) < 1e-3:
        run_type = "HSE06"
    elif abs(aexx - 0.25) < 1e-3 and abs(hf_screen - 0.0) < 1e-3:
        run_type = "PBE0"
    elif abs(aexx - 0.20) < 1e-3:
        run_type = "B3LYP"
    elif lhfcalc:
        # Hybrid but not standard HSE/B3LYP/PBE0
        if hf_screen not in [0.0, 0.2, 0.3]:
            run_type = f"DSH"
        else:
            run_type = "Hybrid (Unspecified)"
    elif incar.get("METAGGA") and incar.get("METAGGA") not in {"--", "None"}:
        run_type = incar.get("METAGGA")
    elif incar.get("GGA"):
        run_type = incar.get("GGA")
    else:
        run_type = "GGA"

    # --- Space group ---
    symbol = sg.get_space_group_symbol()
    number = sg.get_space_group_number()

    return symbol, number, run_type, aexx, hf_screen
#symbol, number, functional, aexx, hf_screen = get_space_group_and_functional(vasprun_file)
#print(f"Space group: {symbol} ({number})")
#print(f"Functional type: {functional}, AEXX = {aexx:.2f}, HFSCREEN = {hf_screen:.2f}")

# ------------ Get KS eigen, bands, ... from vasprun.xml file thanks to pymatgen
def read_eigenvals_from_vasprun(file_path, n_cbm_plus=2, n_vbm_minus=2,
                                kpoints_list=None, verbose=False):
    vrun = Vasprun(file_path, parse_projected_eigen=False)
    bs = vrun.get_band_structure()
    vbm = bs.get_vbm()
    cbm = bs.get_cbm()
    gap = bs.get_band_gap()
   
# --- Extract indices (0-based in pymatgen) ---
    vbm_indices = vbm["band_index"][Spin.up]
    cbm_indices = cbm["band_index"][Spin.up]
    vbm_band_index = vbm_indices[0]
    cbm_band_index = cbm_indices[0]

    vbm_kpoint = [float(round(x, 5)) for x in vbm["kpoint"].frac_coords]
    cbm_kpoint = [float(round(x, 5)) for x in cbm["kpoint"].frac_coords]

    if verbose:
        vbm_bands_str = ", ".join(str(i + 1) for i in vbm_indices)
        cbm_bands_str = ", ".join(str(i + 1) for i in cbm_indices)
        print(f"VBM: {vbm['energy']:.4f} eV at {vbm_kpoint}, band(s) {vbm_bands_str}")
        print(f"CBM: {cbm['energy']:.4f} eV at {cbm_kpoint}, band(s) {cbm_bands_str}")
        print(f"Band gap: {gap['energy']:.4f} eV\n")

    kpoints = vrun.actual_kpoints
    eigenvalues_data = vrun.eigenvalues[Spin.up]

    # Select only requested k-points
    tol = 1e-3
    if kpoints_list:
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

        selected_bands = set()

        # Add VBM ± n_vbm_minus
        for b in range(vbm_band_index - n_vbm_minus, vbm_band_index + 2):
            if 0 <= b < nband_total:
                selected_bands.add(b)

        # Add CBM ± n_cbm_plus (and one below to be safe)
        for b in range(cbm_band_index - 1, cbm_band_index + n_cbm_plus + 1):
            if 0 <= b < nband_total:
                selected_bands.add(b)

        band_indices = sorted(selected_bands)

        eigvals = [float(eigenvalues_data[i][b][0]) for b in band_indices]
        occs = [float(eigenvalues_data[i][b][1]) for b in band_indices]
        bands = [b + 1 for b in band_indices]

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
    n_cbm_plus=4,
    n_vbm_minus=4,
    kpoints_list=[[0.5, 0.5, 0.5]], # R-point for cubic primitive cell
    verbose=False
)

# ----- print some result to compare to my old bash script
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
print("Another test to see format:", list_n, eig_n, occu_n )

# ------------- extract u_nk at k-points and for 6 nbands: cbm-3,2,1 vbm-1,2,3 --------
# link to example: https://hungqpham.com/mcu/plottingwfn.html#parameters-for-get-unk
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
    #vasp = mcu.VASP(vasprun_dir)

    #print("K-POINTS  :", vasp.kpts) 
    #print("! ! ! Shape: g_coeff, g_vecs:", gvec.shape, g_coeff.shape, gvec[0], g_coeff[0][1])
#g_coeff, g_vecs  (5853, 3) (1, 11706) [ 2.2379435e-03-0.00307908j -2.5011737e-05-0.00015305j
#  2.4479514e-04-0.00033802j ... -2.9912220e-02-0.01805255j
#  5.1886704e-02+0.03210271j  6.2322207e-03+0.00032662j]
    return gvec, g_coeff

# -------------- End of functions: lets loop over bands and extract Gves, C_{n,k}(Gvecs}------------------ 
# Gabriel cleaned this part, but it can still be improved --> Lechat:MistraAI
for j, n in enumerate(list_n):

    # get Gvecs, g_coeff for cbm,vbm or band list [], and kpoints
    gvec, g_coeff = extract_wavecar_data(l_wavecar_file, band_list=[n], k_points=0)
    gvec=list(gvec)
    # decompose to real and imag: note that for SOC we have spinors/doubled
    re_gcoeff = list(np.real(g_coeff))[0]
    im_gcoeff = list(np.imag(g_coeff))[0]
    print(re_gcoeff.shape, g_coeff[0].shape)
    #re_gcoeff_down = list(np.real(g_coeff))[1]
    #im_gcoeff_down = list(np.imag(g_coeff))[1]
    # get band and general info here from vasprun.xml 
    Elements, n_atoms, is_soc, n_elec, n_bands, cell_vecs = get_general_info(vasprun_dir)
    # get space group, and functional from vasprun.xml 
    
    symbol, number, functional, aexx, hf_screen = get_space_group_and_functional(vasprun_file)

    if j==0:
        wmode='w'
    else:
        wmode='a'

    with open(output_file, wmode) as f:
        if j==0:
            
            f.write(f"# Space group      : {symbol} ({number})\n")
            # write verbose to output file: to be improved later
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
                    f"# Functional: {functional}, AEXX = {aexx:.2f}, HFSCREEN = {hf_screen:.2f}\n"
                    f"# |----------------- Begin Gvecs and C_nk(Gvecs)-------------------------------|\n"
                    )
            # write main results here: 
            f.write("#\tGx\tGy\tGz\tRe(C_nk)\tIm(C_nk)\n")

        # change range of g_coeffs by 1/2 if lsorbit = true  
        range_g_coeff = len(re_gcoeff)
        range_g_coeff = len(re_gcoeff) // (2 if is_soc else 1)
        if j == 0:
            print(f"Check len(re_gcoeff) : {range_g_coeff}, soc: {is_soc}")

        for i in range(range_g_coeff):
            if i==0:
                text=f'n_band : {n},\tenergy : {eig_n[j]},\toccupation : {occu_n[j]}\n'
                #print("CHECK: NBANDS", n_bands)
            else:
                text=''
            f.write(f"{text}\t{gvec[i][0]}\t{gvec[i][1]}\t{gvec[i][2]}\t{round(float(re_gcoeff[i]), 8):13.8f}     {round(float(im_gcoeff[i]), 8):13.8f}\n")

        f.write('\n')

# Improved version should path to WAVECAR and vasprun here
# TO DO
#   1. avoid using mcu, read WAVECAR directly with pymatgen:
#   2. combine some functions and clean, command line or minimum input file
#   3. write simple k.p model to plot bandstructure, compare to DFT with DSH
#   4. whatever Simon wants
