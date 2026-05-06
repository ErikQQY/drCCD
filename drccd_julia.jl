module DRCCDJulia

using LinearAlgebra
using NonlinearSolve

export solve_drccd_energy

function rhs_drccd(T, ovov, ovvo)
    # T: i,j,a,b
    # ovov: i,a,j,b
    # ovvo: i,a,b,j

    term = 2 .* permutedims(
        @views(einsum_iakj_kjcb(ovvo, T)),
        (1, 3, 2, 4),
    )

    term .+= 2 .* permutedims(
        @views(einsum_ikac_jbck(T, ovvo)),
        (1, 3, 2, 4),
    )

    tmp = einsum_ikac_kcld(T, ovov)          # i,a,l,d
    term .+= 4 .* permutedims(
        einsum_iald_ljdb(tmp, T),
        (1, 3, 2, 4),
    )

    # + ovov.transpose(0,2,1,3) in PySCF: i,j,a,b
    term .+= permutedims(ovov, (1, 3, 2, 4))

    return term
end

# ovvo[i,a,c,k] * T[k,j,c,b] -> i,a,j,b
function einsum_iakj_kjcb(ovvo, T)
    nocc, nvir, _, _ = size(ovvo)
    out = zeros(eltype(T), nocc, nvir, nocc, nvir)
    @inbounds for i in 1:nocc, a in 1:nvir, j in 1:nocc, b in 1:nvir
        s = zero(eltype(T))
        for k in 1:nocc, c in 1:nvir
            s += ovvo[i,a,c,k] * T[k,j,c,b]
        end
        out[i,a,j,b] = s
    end
    return out
end

# T[i,k,a,c] * ovvo[j,b,c,k] -> i,a,j,b
function einsum_ikac_jbck(T, ovvo)
    nocc, nvir, _, _ = size(ovvo)
    out = zeros(eltype(T), nocc, nvir, nocc, nvir)
    @inbounds for i in 1:nocc, a in 1:nvir, j in 1:nocc, b in 1:nvir
        s = zero(eltype(T))
        for k in 1:nocc, c in 1:nvir
            s += T[i,k,a,c] * ovvo[j,b,c,k]
        end
        out[i,a,j,b] = s
    end
    return out
end

# T[i,k,a,c] * ovov[k,c,l,d] -> i,a,l,d
function einsum_ikac_kcld(T, ovov)
    nocc, nvir, _, _ = size(ovov)
    out = zeros(eltype(T), nocc, nvir, nocc, nvir)
    @inbounds for i in 1:nocc, a in 1:nvir, l in 1:nocc, d in 1:nvir
        s = zero(eltype(T))
        for k in 1:nocc, c in 1:nvir
            s += T[i,k,a,c] * ovov[k,c,l,d]
        end
        out[i,a,l,d] = s
    end
    return out
end

# tmp[i,a,l,d] * T[l,j,d,b] -> i,a,j,b
function einsum_iald_ljdb(tmp, T)
    nocc, nvir, _, _ = size(tmp)
    out = zeros(eltype(T), nocc, nvir, nocc, nvir)
    @inbounds for i in 1:nocc, a in 1:nvir, j in 1:nocc, b in 1:nvir
        s = zero(eltype(T))
        for l in 1:nocc, d in 1:nvir
            s += tmp[i,a,l,d] * T[l,j,d,b]
        end
        out[i,a,j,b] = s
    end
    return out
end

function solve_drccd_energy(ovov_py, ovvo_py, mo_energy_py;
                            abstol=1e-10, reltol=1e-10, maxiters=200)
    ovov = Array{Float64}(ovov_py)
    ovvo = Array{Float64}(ovvo_py)
    mo_energy = Array{Float64}(mo_energy_py)

    nocc, nvir, _, _ = size(ovov)

    eia = mo_energy[1:nocc] .- reshape(mo_energy[nocc+1:end], 1, :)
    denom = zeros(Float64, nocc, nocc, nvir, nvir)

    @inbounds for i in 1:nocc, j in 1:nocc, a in 1:nvir, b in 1:nvir
        denom[i,j,a,b] = eia[i,a] + eia[j,b]
    end

    # MP2-like initial guess, same as PySCF init_amps
    T0 = permutedims(ovov, (1, 3, 2, 4)) ./ denom

    function residual!(Rvec, Tvec, p)
        T = reshape(Tvec, nocc, nocc, nvir, nvir)
        R = reshape(Rvec, nocc, nocc, nvir, nvir)
        R .= denom .* T .- rhs_drccd(T, ovov, ovvo)
        return nothing
    end

    prob = NonlinearProblem(
        NonlinearFunction(residual!),
        vec(copy(T0)),
    )

    sol = solve(prob, NewtonRaphson();
                abstol=abstol, reltol=reltol, maxiters=maxiters)

    T = reshape(Array(sol.u), nocc, nocc, nvir, nvir)

    energy = 0.0
    @inbounds for i in 1:nocc, j in 1:nocc, a in 1:nvir, b in 1:nvir
        energy += 2.0 * T[i,j,a,b] * ovov[i,a,j,b]
    end

    return energy, T, sol.retcode
end

end