Getting Started
===============

This guide will help you get started with metalog-jax.

Installation
------------

Install metalog-jax using pip:

.. code-block:: bash

   pip install metalog-jax

Or with uv:

.. code-block:: bash

   uv pip install metalog-jax

Basic Usage
-----------

Fitting a Metalog Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to use metalog-jax is to fit a distribution to your data using
the ``fit`` function from ``metalog_jax.metalog``:

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

   # Evaluate the quantile function (inverse CDF)
   q_75 = m.ppf(jnp.array([0.75]))

   # Compute distribution properties
   mean_val = m.mean
   median_val = m.median
   variance_val = m.var
   std_val = m.std

SPT Metalog (3-Term Closed-Form)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a simpler 3-term metalog with closed-form solutions, use the SPT (Symmetric
Percentile Triplet) variant. SPT uses exactly three quantiles: alpha, 0.5 (median),
and (1 - alpha):

.. code-block:: python

   import jax.numpy as jnp
   from metalog_jax.base import MetalogBoundedness, SPTMetalogParameters
   from metalog_jax.metalog import fit_spt_metalog

   # Generate sample data
   data = jnp.array([1.2, 2.3, 2.8, 3.5, 4.1, 5.6, 6.2, 7.8, 9.1])

   # Configure SPT parameters (uses 10th, 50th, 90th percentiles with alpha=0.1)
   params = SPTMetalogParameters(
       boundedness=MetalogBoundedness.UNBOUNDED,
       alpha=0.1,
       lower_bound=0.0,
       upper_bound=0.0,
   )

   spt = fit_spt_metalog(data, params)

   # Evaluate quantile function
   q_25 = spt.ppf(jnp.array([0.25]))

   # Compute properties
   mean_val = spt.mean
   std_val = spt.std

Using Pre-computed Quantiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have quantile values (e.g., from expert elicitation), use
``precomputed_quantiles=True``:

.. code-block:: python

   import jax.numpy as jnp
   from metalog_jax.base import MetalogInputData, MetalogParameters
   from metalog_jax.base import MetalogBoundedness, MetalogFitMethod
   from metalog_jax.metalog import fit

   # Pre-computed quantiles (must be strictly ascending)
   quantiles = jnp.array([5.0, 10.0, 20.0])
   probability_levels = jnp.array([0.1, 0.5, 0.9])

   # Create input data with precomputed quantiles
   data = MetalogInputData.from_values(
       x=quantiles,
       y=probability_levels,
       precomputed_quantiles=True
   )

   params = MetalogParameters(
       boundedness=MetalogBoundedness.UNBOUNDED,
       method=MetalogFitMethod.OLS,
       lower_bound=0.0,
       upper_bound=0.0,
       num_terms=3
   )

   m = fit(data, params)

Bounded Distributions
---------------------

For data with known bounds, use bounded metalog variants:

.. code-block:: python

   from metalog_jax.base import MetalogInputData, MetalogParameters
   from metalog_jax.base import MetalogBoundedness, MetalogFitMethod
   from metalog_jax.metalog import fit

   # Bounded metalog for data in [0, 100]
   params = MetalogParameters(
       boundedness=MetalogBoundedness.BOUNDED,
       method=MetalogFitMethod.OLS,
       num_terms=5,
       lower_bound=0.0,
       upper_bound=100.0,
   )

   m = fit(data, params)

Boundedness options:

- ``MetalogBoundedness.UNBOUNDED``: Full real line support (-∞, ∞)
- ``MetalogBoundedness.STRICTLY_LOWER_BOUND``: Support on (lower_bound, ∞)
- ``MetalogBoundedness.STRICTLY_UPPER_BOUND``: Support on (-∞, upper_bound)
- ``MetalogBoundedness.BOUNDED``: Support on (lower_bound, upper_bound)

Regression Methods
------------------

metalog-jax supports two regression methods for fitting, organized in the
``metalog_jax.regression`` module:

- **OLS** (``metalog_jax.regression.ols``): No regularization, closed-form solution
- **LASSO** (``metalog_jax.regression.lasso``): L1 regularization for sparse solutions

Ordinary Least Squares (OLS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Default method, no regularization:

.. code-block:: python

   from metalog_jax.base import MetalogInputData, MetalogParameters
   from metalog_jax.base import MetalogBoundedness, MetalogFitMethod
   from metalog_jax.metalog import fit

   metalog_params = MetalogParameters(
       boundedness=MetalogBoundedness.UNBOUNDED,
       method=MetalogFitMethod.OLS,
       lower_bound=0.0,
       upper_bound=0.0,
       num_terms=5
   )

   m = fit(data, metalog_params)

LASSO Regression
~~~~~~~~~~~~~~~~

L1 regularization for sparse solutions using proximal gradient descent:

.. code-block:: python

   from metalog_jax.base import MetalogInputData, MetalogParameters
   from metalog_jax.base import MetalogBoundedness, MetalogFitMethod
   from metalog_jax.metalog import fit
   from metalog_jax.regression.lasso import LassoParameters

   # Configure metalog with Lasso method
   metalog_params = MetalogParameters(
       boundedness=MetalogBoundedness.UNBOUNDED,
       method=MetalogFitMethod.Lasso,
       lower_bound=0.0,
       upper_bound=0.0,
       num_terms=9
   )

   # Specify LASSO hyperparameters
   lasso_params = LassoParameters(
       lam=0.1,               # L1 regularization strength
       learning_rate=0.01,
       num_iters=500,
       tol=1e-6,
       momentum=0.9
   )

   # Fit with LASSO regression
   m = fit(data, metalog_params, regression_hyperparams=lasso_params)

Grid Search with JAX vmap
-------------------------

Use JAX's ``vmap`` for efficient hyperparameter search:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from metalog_jax.base import MetalogInputData, MetalogParameters
   from metalog_jax.base import MetalogBoundedness, MetalogFitMethod
   from metalog_jax.metalog import fit, GridResult
   from metalog_jax.regression.lasso import LassoParameters
   from metalog_jax.utils import DEFAULT_Y, ks_distance

   # Create data
   data = MetalogInputData.from_values(rvs, DEFAULT_Y, False)

   # Define fit function for a single L1 penalty
   def fit_with_lasso_penalty(l1_penalty):
       lasso_params = LassoParameters(lam=l1_penalty)
       lasso_metalog_params = MetalogParameters(
           boundedness=MetalogBoundedness.BOUNDED,
           lower_bound=0,
           upper_bound=1,
           method=MetalogFitMethod.Lasso,
           num_terms=11,
       )
       metalog = fit(data, lasso_metalog_params, lasso_params)
       fitted_quantiles = metalog.ppf(data.y)
       ks_dist = ks_distance(data.x, fitted_quantiles)
       return GridResult(metalog=metalog, ks_dist=ks_dist)

   # Grid search over L1 penalties
   l1_penalties = jnp.array([0.0, 0.01, 0.1, 1.0])
   vmapped_lasso_fit = jax.vmap(fit_with_lasso_penalty)
   lasso_results = vmapped_lasso_fit(l1_penalties)

Goodness of Fit
---------------

Use the Kolmogorov-Smirnov distance to evaluate fit quality:

.. code-block:: python

   import jax.numpy as jnp
   from metalog_jax.utils import ks_distance

   # Two samples from the same distribution
   x = jnp.array([0.1, 0.5, 0.3, 0.7, 0.9])
   y = jnp.array([0.2, 0.4, 0.6, 0.8])
   distance = ks_distance(x, y)

   # Comparing different distributions
   uniform_sample = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
   skewed_sample = jnp.array([0.05, 0.1, 0.15, 0.2, 0.25])
   distance = ks_distance(uniform_sample, skewed_sample)

Serialization
-------------

Save and load fitted metalog distributions:

.. code-block:: python

   from pathlib import Path
   from metalog_jax.metalog import Metalog

   # Save to JSON file
   m.save(Path("my_metalog.json"))

   # Load from JSON file
   loaded = Metalog.load(Path("my_metalog.json"))

   # Or use string serialization
   json_str = m.dumps()
   loaded = Metalog.loads(json_str)

Distribution Methods
--------------------

All metalog distributions (``Metalog`` and ``SPTMetalog``) provide these methods
and properties:

**Methods:**

- ``ppf(y)``: Quantile function (percent point function / inverse CDF)
- ``cdf(q)``: Cumulative distribution function
- ``pdf(x)``: Probability density function
- ``logpdf(x)``: Natural log of the PDF
- ``sf(x)``: Survival function (1 - CDF)
- ``logsf(x)``: Natural log of the survival function
- ``isf(q)``: Inverse survival function
- ``rvs(params)``: Random variable generation
- ``plot(option)``: Visualize the distribution (PDF, CDF, or SF)
- ``save(path)``: Save to JSON file
- ``dumps()``: Serialize to JSON string

**Properties:**

- ``mean``: Expected value
- ``median``: Median (50th percentile)
- ``var``: Variance
- ``std``: Standard deviation
- ``mode``: Most likely value
- ``num_terms``: Number of terms in the metalog expansion
- ``boundedness``: Boundedness type
- ``lower_bound``: Lower bound value
- ``upper_bound``: Upper bound value

**R Compatibility Aliases:**

- ``q(x)``: Alias for quantile function (R metalog compatibility)
- ``p(q)``: Alias for CDF (R metalog compatibility)
- ``d(q)``: Alias for density function (R metalog compatibility)

Module Structure
----------------

The library is organized into the following modules:

- ``metalog_jax.base``: Core classes and enumerations

  - ``metalog_jax.base.core``: MetalogBase class
  - ``metalog_jax.base.data``: MetalogInputData and MetalogBaseData
  - ``metalog_jax.base.enums``: MetalogBoundedness, MetalogFitMethod, MetalogPlotOptions
  - ``metalog_jax.base.parameters``: MetalogParameters, SPTMetalogParameters

- ``metalog_jax.metalog``: Fitting functions and distribution classes

  - ``fit()``: Fit a standard metalog distribution
  - ``fit_spt_metalog()``: Fit an SPT metalog distribution
  - ``Metalog``: Standard metalog class
  - ``SPTMetalog``: SPT metalog class
  - ``GridResult``: Results container for grid search

- ``metalog_jax.regression``: Regression implementations

  - ``metalog_jax.regression.base``: RegressionModel, RegularizedParameters
  - ``metalog_jax.regression.ols``: OLS regression
  - ``metalog_jax.regression.lasso``: LASSO regression

Next Steps
----------

- Explore the :doc:`api/modules` for detailed API documentation
- Check out the examples in the repository
