import jax
import jax.numpy as jnp

def sq_euclidean_dist(x, y):
    assert x.ndim == y.ndim == 1, "sq_euclidean_dist expects two vectors"

    return jnp.sum((x - y) ** 2)

def rbf_kernel(x, sigma):
    return jnp.exp(x / (-2 * sigma ** 2))

def kernel_matrix(x, sigma, dist_fn = sq_euclidean_dist):
    pairwise_dist_fn = jax.vmap(jax.vmap(dist_fn, (0, None)), (None, 0))
    distances = pairwise_dist_fn(x, x)
    kernel = rbf_kernel(distances, sigma)

    return kernel

def global_error(kx, ky, kz, z, gamma, sigma):
    kx_bar = kx - jnp.mean(kx, axis=1)
    ky_bar = ky - jnp.mean(ky, axis=1)
    alpha = jnp.expand_dims(-2 * kz[0] / (sigma ** 2), axis=-1)
    alpha = alpha * (z[0] - z)
    alpha_bar = alpha - jnp.mean(alpha, axis=0)
    xi = jnp.expand_dims(kx_bar[0] - gamma * ky_bar[0], axis=-1) * alpha_bar
    xi = jnp.sum(xi, axis=0)

    return xi
