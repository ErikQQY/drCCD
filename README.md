# drCCD Solver: Julia and Python Integration

This project implements a quantum chemistry simulation with **direct Ring Coupled Cluster Doubles (drCCD)** correlation energy solver in Julia.

## Overview

The project consists of two main components:

### Julia Part: `drccd_julia.jl`

A high-performance drCCD solver implemented in Julia that efficiently solves the drCCD amplitude equation and trturns the correlation energy.

**Key Features:**
- **Main Function**: `solve_drccd_energy(ovov_py, ovvo_py, mo_energy_py; abstol=1e-10, reltol=1e-10, maxiters=200)`
  - Solves the drCCD energy equations using Newton-Raphson iteration
  - Input tensors: `ovov` (occupied-virtual-occupied-virtual) and `ovvo` (occupied-virtual-virtual-occupied) molecular integrals
  - Input array: `mo_energy_py` (molecular orbital energy array from python)
  - Returns: correlation energy, T2 amplitudes (doubles), and solver return code

**Core Computational Functions:**
- `rhs_drccd(T, ovov, ovvo)`: Computes the right-hand side of the coupled cluster equations
- `einsum_iakj_kjcb(ovvo, T)`: Tensor contraction for ovvo ⊗ T → i,a,j,b
- `einsum_ikac_jbck(T, ovvo)`: Tensor contraction for T ⊗ ovvo → i,a,j,b
- `einsum_ikac_kcld(T, ovov)`: Tensor contraction for T ⊗ ovov → i,a,l,d
- `einsum_iald_ljdb(tmp, T)`: Tensor contraction for tmp ⊗ T → i,a,j,b

---

### Python Part: `test.py`

A test script that demonstrates the drCCD solver integration with quantum chemistry calculations.

**Main Workflow:**

1. **Molecule Setup**: Creates an HF (hydrogen-fluoride) molecule using PySCF with cc-pVDZ basis set
2. **SCF Calculation**: Performs restricted Hartree-Fock (RHF) calculation with density fitting
3. **Integral Construction**: Builds necessary molecular integrals:
   - Extracts occupied-virtual (ov) Cholesky vectors from DF-ERI
   - Constructs `ovov` and `ovvo` tensors from Cholesky factorization
   - Extracts Fock matrix diagonal (molecular orbital energies)
4. **drCCD Solver Call**: Invokes Julia solver via `juliacall` package
5. **Validation**: Compares results with RPA (Random Phase Approximation) correlation energy

**Key Function:**
- `build_drccd_integrals(mf)`: Extracts and constructs necessary tensors and energies from PySCF mean-field object
  - Returns: `(ovov, ovvo, mo_energy)` arrays

**Output Information:**
- Julia solver return code (convergence status)
- drCCD correlation energy from Julia
- RPA correlation energy from PySCF
- Numerical difference between methods

---

This will:
1. Build an HF molecule and compute RHF wavefunction
2. Construct molecular integrals
3. Solve drCCD equations using the Julia solver
4. Print convergence status, energies, and comparison with RPA
