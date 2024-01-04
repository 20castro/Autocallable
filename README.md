# Autocallable

The folder *simple_monte_carlo* contains the implementation of a naive Monte-Carlo approach to autocallable pricing under both Black-Scholes and Heston models. Their applications and the coressponding charts are in the notebook *results.ipynb*.

The folder *one_step_survival* implements the algorithm proposed [here](https://www.researchgate.net/publication/267630465_A_Monte_Carlo_pricing_algorithm_for_autocallables_that_allows_for_stable_differentiation). This algorithm allows to compute the $\Delta$ and $\mathcal V$ of the autocallable in a stable way, unlike the usual techniques. It uses Black-Scholes model.

Authors : David Castro and Maxime Leroy
