import numpy as np
from typing import Callable, Tuple, List


def rnd_mv_gaussian_momentum(rng, ndim: int) -> np.ndarray:
    return rng.normal(0, 1, ndim)


def neg_log_mv_gaussian_density(p: np.ndarray) -> float:
    return (p @ p) / 2


def neg_log_mv_gaussian_density_gradient(p: np.ndarray) -> np.ndarray:
    return p


def hamiltonian(
    q: np.ndarray, p: np.ndarray, neg_log_target: Callable[[np.ndarray], float]
) -> float:
    kinetic_e = neg_log_target(q)
    potential_e = neg_log_mv_gaussian_density(p)
    return kinetic_e + potential_e


def hmc(
    init_position: np.ndarray,
    neg_log_target: Callable[[np.ndarray], float],
    neg_log_target_gradient: Callable[[np.ndarray], np.ndarray],
    step_size: float,
    integration_length: int,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng()
    q = init_position.copy()
    init_momentum = rnd_mv_gaussian_momentum(rng, q.shape[0])
    p = init_momentum.copy()
    traj = [(init_position, init_momentum)]
    for i in range(integration_length):
        p -= 0.5 * step_size * neg_log_target_gradient(q)
        q += step_size * neg_log_mv_gaussian_density_gradient(p)
        p -= 0.5 * step_size * neg_log_target_gradient(q)
        traj.append((q.copy(), p.copy()))
    h_prime = hamiltonian(q, p, neg_log_target)
    h_init = hamiltonian(init_position, init_momentum, neg_log_target)
    if h_init >= h_prime:
        return q, traj
    u = rng.uniform(0, 1)
    if u <= np.exp(h_init - h_prime):
        return q, traj
    else:
        return init_position, traj


m = np.asarray([[1, 0.5], [0.5, 1]])
inv_m = np.linalg.inv(m)


def neg_log_gaussian_target(q: np.ndarray) -> float:
    return (q.T @ inv_m @ q) / 2


def neg_log_gaussian_target_gradient(q: np.ndarray) -> np.ndarray:
    return inv_m @ q
