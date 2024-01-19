import jax
import jax.numpy as jnp

from projectlib.utils import grow_dims, grow_to, flatten

def sq_euclidean_dist(x, y):
    assert x.ndim == y.ndim == 1, "sq_euclidean_dist expects two vectors"

    return jnp.sum((x - y) ** 2)

def rbf_kernel(x, sigma):
    return jnp.exp(x / (-2 * sigma ** 2))

def kernel_matrix(x, sigma, dist_fn = sq_euclidean_dist):
    pairwise_dist_fn = jax.vmap(jax.vmap(dist_fn, (0, None)), (None, 0))
    distances = pairwise_dist_fn(x, x)
    sigma = sigma * jnp.sqrt(x.shape[-1])
    kernel = rbf_kernel(distances, sigma)

    return kernel

def hsic_bottleneck(x, y, z, gamma, sigma_x, sigma_y, sigma_z):
    # compute kernel matrices
    Kx = kernel_matrix(flatten(x), sigma_x)
    Ky = kernel_matrix(flatten(y), sigma_y)
    Kz = kernel_matrix(flatten(z), sigma_z)
    # normalization matrix
    nsamples = x.shape[0]
    H = jnp.identity(nsamples) - jnp.ones((nsamples, nsamples)) / nsamples
    # compute HSIC terms
    hsic_x = jnp.trace(Kx @ H @ Kz @ H) / ((nsamples - 1) ** 2)
    hsic_y = jnp.trace(Ky @ H @ Kz @ H) / ((nsamples - 1) ** 2)

    return (1 - gamma) * hsic_x - gamma * hsic_y, hsic_x, hsic_y

def global_error(kx, ky, kz, z, gamma, sigma):
    kx_bar = kx - jnp.mean(kx, axis=1)
    ky_bar = ky - jnp.mean(ky, axis=1)
    sigma = sigma * jnp.sqrt(jnp.prod(jnp.array(z.shape[1:])))
    alpha = grow_to(2 * kz[-1] / (sigma ** 2), z.ndim)
    alpha = alpha * (z[-1] - z)
    N = alpha.shape[0] # samples in batch
    alpha_bar = jnp.sum(alpha[:-1], axis=0) / N
    kxky = grow_to(kx_bar - gamma * ky_bar, alpha.ndim + 1)
    xi = (jnp.trace(kxky[:-1, :-1] * alpha[-1] / N) +
          kxky[-1, -1] * alpha_bar) / ((N - 1) ** 2)
    # xi = grow_to(kx_bar[-1, -1] - gamma * ky_bar[-1, -1], alpha.ndim) * alpha_bar
    # xi += jnp.sum((grow_to(kx_bar[-1, :-1] - gamma * ky_bar[-1, :-1], alpha.ndim) *
    #                (alpha[-1] + alpha_bar)), axis=0)
    # xi += jnp.sum((grow_to(kx_bar[:-1, -1] - gamma * ky_bar[:-1, -1], alpha.ndim) *
    #                alpha[:-1] * (1 - 1 / N)), axis=0)
    # xi += jnp.sum((grow_to(kx_bar[:-1, :-1] - gamma * ky_bar[:-1, :-1], alpha.ndim + 1) *
    #                alpha[:-1] * (-1 / N)), axis=(0, 1))
    # xi = (jnp.sum(kxky[-1, :]) * alpha_bar -
    #       jnp.sum(kxky[:-1, :] * jnp.expand_dims(alpha[:-1] / N, axis=1), axis=(0, 1)) +
    #       jnp.sum(kxky[-1, :-1] * alpha[:-1], axis=0) +
    #       jnp.sum(kxky[:-1, -1] * alpha[:-1], axis=0)) / ((N - 1) ** 2)
        #   jnp.sum(kxky[-1, :-1] * alpha[:-1], axis=0)) / ((N - 1) ** 2)

    return grow_dims(xi, before=1, after=0) # add batch dim back
