import numpy as np
from functools import reduce

from pyscf import gto, scf
from pyscf.cc import ccsd
from pyscf.ao2mo import _ao2mo
from pyscf.gw import rpa

from juliacall import Main as jl

jl.include("drccd_julia.jl")


def build_drccd_integrals(mf):
    cc = ccsd.CCSD(mf)
    mo_coeff = ccsd._mo_without_core(cc, cc.mo_coeff)

    dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
    fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
    fock = reduce(np.dot, (mo_coeff.T, fockao, mo_coeff))

    nocc = cc.nocc
    nmo = cc.nmo
    nvir = nmo - nocc

    mo = cc._scf.mo_coeff
    Lpq = cc._scf._cderi

    ijslice = (0, nocc, nocc, nmo)
    Lov = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym="s2", mosym="s1")

    eris_ovov = Lov.T @ Lov
    ovov = eris_ovov.reshape(nocc, nvir, nocc, nvir)
    ovvo = ovov.transpose(0, 1, 3, 2).copy()

    mo_energy = fock.diagonal().real
    return ovov, ovvo, mo_energy


mol = gto.Mole()
mol.verbose = 4
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "ccpvdz"
mol.build()

mf = scf.RHF(mol).density_fit()
mf.kernel()

ovov, ovvo, mo_energy = build_drccd_integrals(mf)

e_drccd_julia, t2_julia, retcode = jl.DRCCDJulia.solve_drccd_energy(
    ovov, ovvo, mo_energy,
)

myrpa = rpa.RPA(mf)
e_rpa = myrpa.kernel()

print("Julia solver retcode:", retcode)
print("drCCD correlation energy from Julia:", float(e_drccd_julia))
print("RPA correlation energy from PySCF:", e_rpa)
print("Difference between Julia drCCD and PySCF dRPA:", float(e_drccd_julia) - e_rpa)