from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn.initializers as init

import equinox as eqx
from e3nn_jax import bessel


w_init = init.glorot_uniform()   
pangle = lambda D: jnp.arctan2(
    jnp.sqrt(jnp.sum(jnp.cross(D[:, None, :], D[None, :, :])**2 + 1e-11, axis=2)),
    D @ D.T,
)


def full_graph_masks(N):
    """Mask out self-loops only."""
    idx = jnp.arange(N)
    # Returns a boolean mask: [N, N]
    return ~(idx[:, None] == idx[None, :])


class MorphoEGNN(eqx.Module):
    # Fixed parameters
    num_genes: int
    num_radial_basis: int
    use_angle: bool
    use_attention: bool
    use_rest_length: bool
    use_dropout: bool
    r_cutoff: Optional[float]
    
    # Learnable parameters (Equinox Modules/Arrays)
    phi_e: eqx.Module 
    phi_x: eqx.Module 
    phi_h: eqx.Module
    phi_l: eqx.Module
    phi_attn: eqx.Module
    dropout: eqx.Module
    
    def __init__(
        self, 
        num_heads: int,
        num_genes: int, 
        hidden_dim: int, 
        num_layers: int, 
        num_radial_basis: int,
        use_angle: bool,
        use_dropout: bool,
        use_attention: bool,
        use_rest_length: bool,
        rate: float,
        key: jr.PRNGKey,
        r_cutoff: Optional[float] = None, 
    ):

        keys = jr.split(key, 7)

        self.num_genes = num_genes
        self.num_radial_basis = num_radial_basis
        self.use_angle = use_angle
        self.r_cutoff = r_cutoff
        self.use_dropout = use_dropout
        self.use_attention = use_attention
        self.use_rest_length = use_rest_length
        self.dropout = eqx.nn.Dropout(p=rate)

        # edge MLP: [G_i, G_j, dist²] → hidden_dim
        edge_in_size = 2*num_genes + 1

        if num_radial_basis > 0:
            edge_in_size = edge_in_size - 1 + num_radial_basis
        if use_angle:
            edge_in_size += 1

        self.phi_e = eqx.nn.MLP(
            in_size=edge_in_size,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=num_layers,
            activation=jax.nn.silu,
            key=keys[2],
        )

        # coord‐update MLP: [hidden_dim] → 1
        phi_x_in_size = hidden_dim

        self.phi_x = eqx.nn.MLP(
            in_size=phi_x_in_size,
            out_size=1,
            width_size=hidden_dim,
            depth=num_layers,
            activation=jax.nn.silu,
            key=keys[3],
        )

        # feature update MLP: [G_i, M_i] → num_genes
        phi_h_in_size = num_genes + hidden_dim

        self.phi_h = eqx.nn.MLP(
            in_size=phi_h_in_size,
            out_size=num_genes + 3,
            width_size=hidden_dim,
            depth=num_layers,
            activation=jax.nn.silu,
            key=keys[4],
        )

        self.phi_attn = eqx.nn.MLP(
            in_size=edge_in_size,
            out_size=num_heads,
            width_size=hidden_dim,
            depth=num_layers,
            activation=jax.nn.silu,
            key=keys[5],
        )

        self.phi_l = eqx.nn.MLP(
            in_size=2*num_genes,
            out_size=1,
            width_size=hidden_dim,
            depth=num_layers,
            activation=jax.nn.silu,
            key=keys[6],
        )


    def _mat_mul(self, A, B, in_axes):
        return jax.vmap(lambda A, B: A @ B, in_axes=in_axes)(A, B)

    def __call__(self, t, _X, key, training=None):
        N = _X.shape[0]
        X = _X[:, :3]                    # coords
        G = _X[:, 3:3+self.num_genes]    # gene features
        N = X.shape[0]

        if self.use_dropout:
            G = self.dropout(G, key=key, inference=not training)
        
        # compute pairwise diffs and distances
        diff    = X[:, None, :] - X[None, :, :]    # [N, N, 3]
        dist    = jnp.sum(diff**2, axis=-1)        # [N, N]
        mask    = full_graph_masks(N)              # [N, N]

        # optionally zero‐out beyond cutoff r:
        if self.r_cutoff is not None:
            mask = mask & (dist < self.r_cutoff)

        # build edge features [G_i, G_j, dist²]
        Gi = jnp.tile(G[:, None, :], (1, N, 1))
        Gj = jnp.tile(G[None, :, :], (N, 1, 1))

        if self.use_rest_length:
            l_in = jnp.concatenate([Gi, Gj], axis=-1)  # [N, N, 2*num_genes]
            l_ij = jax.vmap(self.phi_l)(l_in.reshape(-1, l_in.shape[-1])).reshape(N, N, -1)
            term = (1. - l_ij[None, ...] / jnp.sqrt(dist+1e-8).reshape(1, N, N, 1))

        if self.num_radial_basis > 0:
            dist_in = bessel(dist, self.num_radial_basis, 10.)
        else:
            dist_in = dist[..., None]
                             
        e_in = jnp.concatenate([Gi, Gj, dist_in], axis=-1)  # [N, N, 2*num_genes+1]
        if self.use_angle:
            P = _X[:, 3+self.num_genes:]
            e_in = jnp.concatenate((e_in, pangle(P)[..., None]), axis=-1)


        # 1) edge embeddings
        m_ij = jax.vmap(self.phi_e)(e_in.reshape(-1, e_in.shape[-1]))
        m_ij = m_ij.reshape(N, N, -1)    # [N, N, hidden_dim]

        # attention
        if self.use_attention:
            logits = jax.vmap(self.phi_attn)(e_in.reshape(-1, e_in.shape[-1]))
            logits = logits.reshape(N, N, -1)
            logits = jnp.transpose(logits, (2, 0, 1))
            logits = jnp.where(mask[None, ...], logits, -1e9)
            attn_weights = jax.nn.softmax(logits, axis=-1)[..., None]

        # 2) distance‐based scalars for coord update
        w_x = jax.vmap(self.phi_x)(m_ij.reshape(-1, m_ij.shape[-1]))
        w_x = w_x.reshape(1, N, N, 1)

        # 3) coordinate messages
        msg_x = w_x * diff[None, ...]        # [1, N, N, 3]
        if not self.use_rest_length:
            msg_x /= (jnp.sqrt(dist+1e-8).reshape(1, N, N, 1) + 1e-11)
        else:
            msg_x *= term 

        if self.use_attention:
            msg_x_attn = jnp.mean(attn_weights * msg_x, axis=0)   #[N, N, 3]
        else:
            msg_x_attn = jnp.mean(mask.reshape(1, N, N, 1) * msg_x, axis=0)

        if not self.use_rest_length:
            delta_X = jnp.mean(msg_x_attn, axis=1) # [N, 3]
        else:
            if self.use_attention:
                delta_X = jnp.sum(msg_x_attn, axis=1)
            else:
                delta_X = jnp.mean(msg_x_attn, axis=1)

        # 4) feature messages
        if self.use_attention:
            m_ij_attn = jnp.mean(attn_weights * m_ij[None, ...], axis=0)  #[N, N, hidden_dim]
        else:
            m_ij_attn = jnp.mean(mask.reshape(1, N, N, 1) * m_ij[None, ...], axis=0)  #[N, N, hidden_dim]
        m_i_attn = jnp.sum(m_ij_attn, axis=1)

        h_in = jnp.concatenate((G, m_i_attn), axis=-1)

        delta_feature = jax.vmap(self.phi_h)(h_in)

        if not self.use_angle:
            delta_feature = delta_feature.at[:, -3:].set(0.0)
        
        # combine and return
        return jnp.concatenate([delta_X, delta_feature], axis=-1)