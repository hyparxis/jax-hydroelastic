# jax-hydroelastic
Jax implementation of Drake's hydroelastic contact model


* https://arxiv.org/abs/1904.11433
* https://arxiv.org/abs/2110.04157
* https://arxiv.org/abs/1909.05700

TODO: docstrings (decide on a style and enforce it with ruff), make code fuly functional (jit-compatible)

Misc note: while the pure functional style is great the static shape requirement makes working with JAX kind of suck. A DSL like Taichi might provide a better level of abstraction. The downside with Taichi is that we lose the pure functional style that makes JAX beautiful
