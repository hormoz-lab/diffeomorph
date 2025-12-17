import jax.numpy as jnp
import jax.random as jr


triad = jnp.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])


def random_rotation_matrix(key):
    # Split key for axis and angle
    key_axis, key_theta = jr.split(key)

    # Random unit vector as axis
    axis = jr.normal(key_axis, (3,))
    axis = axis / jnp.linalg.norm(axis)

    # Random angle
    theta = jr.uniform(key_theta, (), minval=0.0, maxval=2*jnp.pi)

    # Rodrigues' rotation formula
    K = jnp.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    R = jnp.eye(3) + jnp.sin(theta)*K + (1-jnp.cos(theta))*(K @ K)
    return R


def get_gene_expressions(
    positions, 
    num_genes,
    num_organizer_cells, 
    num_organizer_groups, 
    key,
):

    def get_surface_neighbor_at_pole(idx, dist_sorted_index, is_surface):
        neighbors = dist_sorted_index[idx]
        surface_neighbors = neighbors[is_surface[neighbors]]
        return surface_neighbors[:num_organizer_cells]

    diff = positions[:, None, :] - positions[None, :, :]
    dist = jnp.sum(diff**2, axis=-1) 
    dist_sorted_index = jnp.argsort(dist, axis=1)  

    radial_dist = jnp.linalg.norm(positions, axis=1)
    is_surface = radial_dist > 0.99

    directions = triad @ random_rotation_matrix(key).T
    projections = positions @ directions
    polar_indices_all = jnp.argsort(projections, axis=0)[0]     

    polar_indices = polar_indices_all[:num_organizer_groups]  

    organizer_groups = [
        get_surface_neighbor_at_pole(idx, dist_sorted_index, is_surface) 
        for idx in polar_indices
    ]


    num_cells = positions.shape[0]
    all_indices = jnp.arange(num_cells)
    organizer_indices = jnp.concatenate(organizer_groups)
    rest = all_indices[~jnp.isin(all_indices, organizer_indices)]
    
    # Gene assignment
    genes = jnp.zeros((num_cells, num_genes)) 
    for k, idx in enumerate(organizer_groups):
        genes = genes.at[idx, k].set(1.0)
    
    # Assign last gene channel to all remaining cells
    genes = genes.at[rest, -1].set(1.0)

    return genes, directions, organizer_groups


def prepare_initial_states(
    positions, 
    num_genes,
    elongation_factor,
    num_organizer_cells, 
    num_organizer_groups, 
    key,
):
    
    num_cells = positions.shape[0]

    keys = jr.split(key)
    
    genes, organizer_directions, organizer_groups = get_gene_expressions(
        positions, num_genes, num_organizer_cells, num_organizer_groups, keys[0],
    )
    
    directors = jr.normal(keys[1], (1, 3))
    directors /= jnp.linalg.norm(directors, axis=1, keepdims=True)
    directors = jnp.tile(directors, (num_cells, 1))

    elongation_direction = organizer_directions[:, 0]
    projection = positions @ elongation_direction
    positions += elongation_factor * projection[:, None] * elongation_direction[None, :]

    initial_states = jnp.concatenate(
        (positions, genes, directors), axis=1,
    )

    return initial_states, organizer_groups
