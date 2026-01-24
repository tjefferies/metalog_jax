"""Demonstrates basic Metalog fitting."""

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import tempfile
    from pathlib import Path

    import jax.numpy as jnp
    import numpy as np
    from scipy.stats import t

    from metalog_jax.base import (
        MetalogBoundedness,
        MetalogFitMethod,
        MetalogInputData,
        MetalogParameters,
        MetalogPlotOptions,
        MetalogRandomVariableParameters,
    )
    from metalog_jax.metalog import (
        Metalog,
        fit,
    )
    from metalog_jax.utils import (
        DEFAULT_Y,
        HDRPRNGParameters,
        JaxUniformDistributionParameters,
    )

    return (
        DEFAULT_Y,
        HDRPRNGParameters,
        JaxUniformDistributionParameters,
        Metalog,
        MetalogBoundedness,
        MetalogFitMethod,
        MetalogInputData,
        MetalogParameters,
        MetalogPlotOptions,
        MetalogRandomVariableParameters,
        Path,
        fit,
        jnp,
        np,
        t,
        tempfile,
    )


@app.cell
def _(DEFAULT_Y):
    # default quantiles used to fit metalog, overrideable by user
    DEFAULT_Y
    return


@app.cell
def _(DEFAULT_Y, jnp, t):
    # input data
    dist = t(df=6, loc=50, scale=5)
    rvs = jnp.array(dist.rvs(size=1000, random_state=42))
    ppf = jnp.array(dist.ppf(DEFAULT_Y))
    return ppf, rvs


@app.cell
def _(DEFAULT_Y, MetalogInputData, ppf):
    # init MetalogInputData object from quantiles
    # MUST use `from_values` - performs validity checks
    data = MetalogInputData.from_values(x=ppf, y=DEFAULT_Y, precomputed_quantiles=True)  # noqa: F841
    return


@app.cell
def _(DEFAULT_Y, MetalogInputData, rvs):
    # init MetalogInputData object from raw data
    # MUST use `from_values` - performs validity checks
    data_1 = MetalogInputData.from_values(
        x=rvs, y=DEFAULT_Y, precomputed_quantiles=False
    )
    return (data_1,)


@app.cell
def _(MetalogInputData):
    print(MetalogInputData.__doc__)
    return


@app.cell
def _(MetalogBoundedness, MetalogFitMethod, MetalogParameters):
    # init MetalogParameters
    metalog_params = MetalogParameters(
        boundedness=MetalogBoundedness.UNBOUNDED,
        lower_bound=0,
        upper_bound=1,
        method=MetalogFitMethod.OLS,
        num_terms=5,
    )
    return (metalog_params,)


@app.cell
def _(MetalogParameters):
    print(MetalogParameters.__doc__)
    return


@app.cell
def _(MetalogBoundedness):
    # boundedness options
    list(MetalogBoundedness.__members__.keys())
    return


@app.cell
def _(MetalogFitMethod):
    # fit options
    list(MetalogFitMethod.__members__.keys())
    return


@app.cell
def _(data_1, fit, metalog_params):
    # fit metalog, return distribution object
    metalog = fit(data_1, metalog_params)
    return (metalog,)


@app.cell
def _(fit):
    print(fit.__doc__)
    return


@app.cell
def _(DEFAULT_Y, metalog):
    # get quantiles from fitted metalog
    metalog.ppf(DEFAULT_Y)
    return


@app.cell
def _(Metalog):
    print(Metalog.ppf.__doc__)
    return


@app.cell
def _(DEFAULT_Y, metalog):
    # get quantiles from fitted metalog, compatible with rmetalog API
    metalog.q(DEFAULT_Y)
    return


@app.cell
def _(DEFAULT_Y, metalog, np):
    # results are identical
    ppf_1 = metalog.ppf(DEFAULT_Y)
    q = metalog.q(DEFAULT_Y)
    np.testing.assert_allclose(ppf_1, q)
    return (ppf_1,)


@app.cell
def _(DEFAULT_Y, metalog):
    # can also get logppf
    metalog.logppf(DEFAULT_Y)
    return


@app.cell
def _(Metalog):
    print(Metalog.logppf.__doc__)
    return


@app.cell
def _(DEFAULT_Y, metalog):
    # survival function
    metalog.sf(DEFAULT_Y)
    return


@app.cell
def _(Metalog):
    print(Metalog.sf.__doc__)
    return


@app.cell
def _(DEFAULT_Y, metalog):
    # inverse survival function
    metalog.isf(metalog.sf(DEFAULT_Y))
    return


@app.cell
def _(Metalog):
    print(Metalog.isf.__doc__)
    return


@app.cell
def _(metalog, ppf_1):
    # get the percentile from an input value
    metalog.cdf(ppf_1)
    return


@app.cell
def _(Metalog):
    print(Metalog.cdf.__doc__)
    return


@app.cell
def _(DEFAULT_Y, metalog):
    # to get density of a percentile
    metalog.pdf(DEFAULT_Y)
    return


@app.cell
def _(Metalog):
    print(Metalog.pdf.__doc__)
    return


@app.cell
def _(DEFAULT_Y, metalog):
    # can also generate logpdf
    metalog.logpdf(DEFAULT_Y)
    return


@app.cell
def _(Metalog):
    print(Metalog.logpdf.__doc__)
    return


@app.cell
def _(JaxUniformDistributionParameters):
    # to generate samples from distribution, we first init a PRNG parameter object
    # below is standard jax PRNG
    uniform_prng_params = JaxUniformDistributionParameters(seed=42)
    uniform_prng_params
    return (uniform_prng_params,)


@app.cell
def _(JaxUniformDistributionParameters):
    print(JaxUniformDistributionParameters.__doc__)
    return


@app.cell
def _(HDRPRNGParameters):
    # below is Hubbard Decision Research PRNG
    hdr_prng_params = HDRPRNGParameters()
    hdr_prng_params
    return


@app.cell
def _(HDRPRNGParameters):
    print(HDRPRNGParameters.__doc__)
    return


@app.cell
def _(MetalogRandomVariableParameters, uniform_prng_params):
    # init metalog RV parameters
    metalog_rv_params = MetalogRandomVariableParameters(
        prng_params=uniform_prng_params, size=10
    )
    return (metalog_rv_params,)


@app.cell
def _(MetalogRandomVariableParameters):
    print(MetalogRandomVariableParameters.__doc__)
    return


@app.cell
def _(metalog, metalog_rv_params):
    # generate samples
    metalog.rvs(metalog_rv_params)
    return


@app.cell
def _(Metalog):
    print(Metalog.rvs.__doc__)
    return


@app.cell
def _(metalog):
    # get mean
    metalog.mean
    return


@app.cell
def _(Metalog):
    print(Metalog.mean.__doc__)
    return


@app.cell
def _(metalog):
    # get standard deviation
    metalog.std
    return


@app.cell
def _(Metalog):
    print(Metalog.std.__doc__)
    return


@app.cell
def _(metalog):
    # get variance
    metalog.var
    return


@app.cell
def _(Metalog):
    print(Metalog.var.__doc__)
    return


@app.cell
def _(metalog):
    # get median
    metalog.median
    return


@app.cell
def _(Metalog):
    print(Metalog.median.__doc__)
    return


@app.cell
def _(metalog):
    # get mode
    metalog.mode
    return


@app.cell
def _(Metalog):
    print(Metalog.mode.__doc__)
    return


@app.cell
def _(MetalogPlotOptions):
    # plot options
    list(MetalogPlotOptions.__members__.keys())
    return


@app.cell
def _(MetalogPlotOptions, metalog):
    # plot the Probability Density Function
    metalog.plot(MetalogPlotOptions.PDF)
    return


@app.cell
def _(MetalogPlotOptions, metalog):
    # plot the Cumulative Distribution Function
    metalog.plot(MetalogPlotOptions.CDF)
    return


@app.cell
def _(MetalogPlotOptions, metalog):
    # plot the Survival Function
    metalog.plot(MetalogPlotOptions.SF)
    return


@app.cell
def _(Metalog):
    print(Metalog.plot.__doc__)
    return


@app.cell
def _(Path, metalog, tempfile):
    # save a fitted metalog instance
    with tempfile.TemporaryDirectory() as _tmpdir:
        _save_path = Path(_tmpdir) / "test_metalog.json"
        metalog.save(_save_path)
        assert _save_path.exists()
    return


@app.cell
def _(Metalog):
    print(Metalog.save.__doc__)
    return


@app.cell
def _(Metalog, Path, metalog, np, tempfile):
    # load a previously saved metalog
    with tempfile.TemporaryDirectory() as _tmpdir:
        _save_path = Path(_tmpdir) / "test_metalog.json"
        metalog.save(_save_path)
        loaded_metalog = Metalog.load(_save_path)
        np.testing.assert_array_almost_equal(
            np.array(loaded_metalog.a), np.array(metalog.a)
        )
    return


@app.cell
def _(Metalog):
    print(Metalog.load.__doc__)
    return


if __name__ == "__main__":
    app.run()
