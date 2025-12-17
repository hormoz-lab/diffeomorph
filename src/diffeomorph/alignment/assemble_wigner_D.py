import jax
import jax.numpy as jnp
from jax.lax import cond
from e3nn_jax._src.so3 import change_basis_real_to_complex


def wigner_D_custom(q, ell, mp, m, offset, binoms, prefactors):

    def _rho_term(Ra, Rb, rho):
        _Ra = cond(
            jnp.abs(Ra) > 0,
            lambda _: Ra,
            lambda _: Ra + 1e-10,
            operand=None,
        )
        _Rb = cond(
            jnp.abs(Rb) > 0,
            lambda _: Rb,
            lambda _: Rb + 1e-10,
            operand=None,
        )
        res = ( 
            binoms[ell, mp+offset, m+offset, rho] * 
            jnp.power(_Ra, ell+mp-rho) *
            jnp.power(_Ra.conj(), ell-rho-m) *
            jnp.power(_Rb, rho-mp+m) *
            jnp.power(_Rb.conj(), rho) 
        )
        return res

    Ra = q[0] + 1j * q[3]
    Rb = q[2] + 1j * q[1]
    prefactor = prefactors[ell, mp+offset, m+offset]
    sum_over_rho = jnp.nan_to_num(
        jax.vmap(
            _rho_term,
            in_axes=(None, None, 0),
        )(Ra, Rb, jnp.arange(21))
    ).sum()
    return sum_over_rho * prefactor


def assemble_real_wigner_D_direct(ell_max, q, offset, binoms, prefactors, perms):
    wigner_Ds = jnp.zeros((ell_max, 21, 21))
    for i, ell in enumerate(range(1, ell_max+1)):
        mp_vals = m_vals = jnp.arange(-ell, ell+1)
        Q = change_basis_real_to_complex(ell)
        P = perms[i]
        wigner_D = jax.vmap(
            jax.vmap(        
                lambda mp, m: wigner_D_custom(q, ell, mp, m, offset, binoms, prefactors),
                in_axes=(None, 0),
            ),
            in_axes=(0, None),
        )(mp_vals, m_vals)
        wigner_D_real = P @ jnp.real(Q.T.conj() @ wigner_D @ Q) @ P.T
        wigner_Ds = wigner_Ds.at[i, :2*ell+1, :2*ell+1].set(wigner_D_real)
    return wigner_Ds
