# hmc-first-steps

This is a repo with a Hamiltonian Monte Carlo implementation for educational purposes.

`hmc_mv_with_gaps.py` has code with gaps that need to be completed in order to get a working implementation.
`hmc.py` has a univariate implementation of naive HMC.
`hmc_mv.py` has a multivariate implementation of naive HMC.
`hmc_vis.ipynb` shows how to use the code and generate plots.

I suggest to play around with the code and work on the following exercises:

1. Implement HMC (or repair HMC in `hmc_mv_with_gaps.py`)
2. Attempt sampling from 2D Gaussian with non-diagonal covariance:
  - choose step size (e.g. 0.1)
  - Find suitable integration length by computing distribution of lengths at which U-Turns occur
  - What is your acceptance rate?
  - How many samples do you need until the squared error of the inferred covariance matrix is < 0.05 ?
  - Repeat with simple Metropolis-Hastings.

If you want to learn more, a great starting point is [“MCMC using Hamiltonian Dynamics”, Neil (2012)](https://www.mcmchandbook.net/HandbookChapter5.pdf)
and also [Betancourt 2017](https://arxiv.org/abs/1701.02434)


