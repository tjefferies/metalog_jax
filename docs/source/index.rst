metalog-jax Documentation
=========================

A JAX implementation of the Metalog distribution for flexible probability modeling.

The Metalog distribution is a highly flexible continuous probability distribution
that can be fit to virtually any dataset using quantile-based regression methods.
This library provides efficient JAX-based implementations with automatic
differentiation support.

Features
--------

- **Multiple regression methods**: OLS and LASSO
- **JAX integration**: Full support for JIT compilation and autodiff
- **Grid search**: Hyperparameter optimization for regularized methods
- **Flexible fitting**: Support for bounded and unbounded distributions

Quick Example
-------------

.. code-block:: python

   import jax.numpy as jnp
   from metalog_jax.base import MetalogInputData, MetalogParameters
   from metalog_jax.base import MetalogBoundedness, MetalogFitMethod
   from metalog_jax.metalog import fit

   # Create validated input data using from_values()
   raw_data = jnp.array([1.2, 2.3, 2.8, 3.5, 4.1, 5.6])
   data = MetalogInputData.from_values(
       x=raw_data,
       y=jnp.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95]),
       precomputed_quantiles=False  # Raw samples, not precomputed quantiles
   )

   # Configure metalog parameters with OLS
   metalog_params = MetalogParameters(
       boundedness=MetalogBoundedness.UNBOUNDED,
       method=MetalogFitMethod.OLS,
       lower_bound=0.0,
       upper_bound=0.0,
       num_terms=5
   )

   # Fit the metalog distribution
   m = fit(data, metalog_params)

   # Generate quantiles
   quantiles = m.ppf(jnp.array([0.25, 0.5, 0.75]))

   # Access distribution properties
   print(f"Mean: {m.mean}, Median: {m.median}, Std: {m.std}")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting-started
   basic_usage
   fitting_grids

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
