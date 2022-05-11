import numpy as np
from typing import Callable, Tuple, List


def rnd_uv_gaussian_momentum(rng) -> float:
    return rng.normal(0, 1)


def neg_log_uv_gaussian_density(p: float) -> float:
    return (p * p) / 2


def neg_log_uv_gaussian_density_gradient(p: float) -> float:
    return p


def hamiltonian(q: float, p: float, neg_log_target: Callable[[float], float]) -> float:
    kinetic_e = neg_log_target(q)
    potential_e = neg_log_uv_gaussian_density(p)
    return kinetic_e + potential_e


def hmc(
    init_position: float,
    neg_log_target: Callable[[float], float],
    neg_log_target_gradient: Callable[[float], float],
    step_size: float,
    integration_length: int,
) -> Tuple[float, List[Tuple[float, float]]]:
    rng = np.random.default_rng()
    q = init_position
    init_momentum = rnd_uv_gaussian_momentum(rng)
    p = init_momentum
    traj = [(init_position, init_momentum)]
    for i in range(integration_length):
        p -= 0.5 * step_size * neg_log_target_gradient(q)
        q += step_size * neg_log_uv_gaussian_density_gradient(p)
        p -= 0.5 * step_size * neg_log_target_gradient(q)
        traj.append((q, p))
    h_prime = hamiltonian(q, p, neg_log_target)
    h_init = hamiltonian(init_position, init_momentum, neg_log_target)
    if h_init >= h_prime:
        return q, traj
    u = rng.uniform(0, 1)
    if u <= np.exp(h_init - h_prime):
        return q, traj
    else:
        return init_position, traj


def neg_log_gaussian_target(q: float) -> float:
    return (q * q) / 2


def neg_log_gaussian_target_gradient(q: float) -> float:
    return q
