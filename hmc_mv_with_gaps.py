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
    # fill in computation of hamiltonian
    return


def hmc(
    init_position: np.ndarray,
    neg_log_target: Callable[[np.ndarray], float],
    neg_log_target_gradient: Callable[[np.ndarray], np.ndarray],
    step_size: float,
    integration_length: int,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng()
    q = init_position.copy()
    # fill in generation of momentum
    # init_momentum =
    p = init_momentum.copy()
    traj = [(init_position, init_momentum)]
    for i in range(integration_length):
        # fill in integration scheme
        traj.append((q.copy(), p.copy()))
    # fill in metropolis acceptance step
    return


inv_m = np.asarray([[1, 0.5], [0.5, 1]])


def neg_log_gaussian_target(q: np.ndarray) -> float:
    return (q.T @ inv_m @ q) / 2


def neg_log_gaussian_target_gradient(q: np.ndarray) -> np.ndarray:
    return inv_m @ q
