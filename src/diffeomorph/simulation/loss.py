import jax
import jax.numpy as jnp

from diffeomorph.alignment import (
    wigner_D,
    solve_inner_alignment_problem,
)


def shape_matching_loss(
    q,
    Z_preds, 
    Z_targets, 
    use_weighted, 
    l_max,
    offset,
    binoms,
    prefactors,
    perms,
    lr,
    tol,
):

    def _l2_loss_with_zernike_moments(q, Z_pred, Z_target):
        wigner_Ds = wigner_D(
            l_max, 
            q, 
            offset, 
            binoms, 
            prefactors, 
            perms,
        )
        res = jax.vmap(
            lambda z_pred, z_target, wigner_D: jnp.sum(
                ((z_pred @ wigner_D.T) - z_target)**2),
            in_axes=(0, 0, 0),
        )(Z_pred[1:], Z_target[1:], wigner_Ds[0:]).sum()
        return res

    if use_weighted:
        Z_pred = Z_preds['weighted']
        Z_target = Z_targets['weighted']
    else:
        Z_pred = Z_preds['unweighted']
        Z_target = Z_targets['unweighted']          

    optimized_q = solve_inner_alignment_problem(
        lr, 
        tol, 
        q, 
        offset, 
        binoms, 
        prefactors, 
        perms,
        l_max, 
        Z_pred,
        Z_target,
    )

    recon_err = _l2_loss_with_zernike_moments(
        optimized_q, Z_pred, Z_target,
    )   

    return recon_err, optimized_q