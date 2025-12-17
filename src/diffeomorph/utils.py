import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from e3nn_jax import Irreps, IrrepsArray


def check_is_vector(irreps):
    if irreps.num_irreps != 1:
        raise ValueError("Input must be a single vector (1x1o or 1x1e).")
    [(mul, ir)] = irreps
    if not (mul == 1 and ir.l == 1):
        raise ValueError("Input must be a vector (1x1o or 1x1e).")
    return ir.p


def get_irreps(X, l_max):
    _X = IrrepsArray('1o', X)
    vec_p = check_is_vector(_X.irreps)
    irreps_out = Irreps([(1, (l, vec_p**l)) for l in range(l_max+1)])    
    return irreps_out



def read_off_file(filepath):
    with open(filepath, 'r') as file:
        header = file.readline().strip()
        if header != "OFF":
            raise ValueError("File is not a valid OFF file.")
        
        counts = file.readline().strip().split()
        n_vertices, n_faces, _ = map(int, counts)
        
        vertices = []
        for _ in range(n_vertices):
            vertex = list(map(float, file.readline().strip().split()))
            vertices.append(vertex)
        
        faces = []
        for _ in range(n_faces):
            face = list(map(int, file.readline().strip().split()))
            faces.append(face[1:])  
        
    return vertices, faces


## Structure generation
def generate_uniform_sphere(N, rng):
    key_phi, key_cos_theta, key_r = jrandom.split(rng, 3)

    phi = jrandom.uniform(key_phi, shape=(N,), minval=0.0, maxval=2 * jnp.pi)
    cos_theta = jrandom.uniform(key_cos_theta, shape=(N,), minval=-1.0, maxval=1.0)
    theta = jnp.arccos(cos_theta)
    
    r = jnp.cbrt(jrandom.uniform(key_r, shape=(N,), minval=0.0, maxval=1.0))
    
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    
    return jnp.vstack((x, y, z)).T


def bend_ellipsoid(points, R=2.0):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    theta = x / R
    x_new = R * np.sin(theta)
    z_new = R * (1 - np.cos(theta)) + z
    return np.vstack((x_new, y, z_new)).T