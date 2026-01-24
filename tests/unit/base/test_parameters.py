# Copyright: Travis Jefferies 2026
"""Tests for metalog_jax.base.parameters module."""

import unittest


class TestParametersImport(unittest.TestCase):
    """Test that parameter classes can be imported from the new module location."""

    def test_import_metalog_random_variable_parameters(self):
        """Test MetalogRandomVariableParameters can be imported."""
        from metalog_jax.base.parameters import MetalogRandomVariableParameters

        self.assertTrue(callable(MetalogRandomVariableParameters))

    def test_import_metalog_parameters_base(self):
        """Test MetalogParametersBase can be imported."""
        from metalog_jax.base.parameters import MetalogParametersBase

        self.assertTrue(callable(MetalogParametersBase))

    def test_import_metalog_parameters(self):
        """Test MetalogParameters can be imported."""
        from metalog_jax.base.parameters import MetalogParameters

        self.assertTrue(callable(MetalogParameters))

    def test_import_spt_metalog_parameters(self):
        """Test SPTMetalogParameters can be imported."""
        from metalog_jax.base.parameters import SPTMetalogParameters

        self.assertTrue(callable(SPTMetalogParameters))


class TestMetalogParametersBase(unittest.TestCase):
    """Tests for MetalogParametersBase class."""

    def test_instantiation(self):
        """Test MetalogParametersBase can be instantiated."""
        from metalog_jax.base.enums import MetalogBoundedness
        from metalog_jax.base.parameters import MetalogParametersBase

        params = MetalogParametersBase(
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
        )

        self.assertEqual(params.boundedness, MetalogBoundedness.UNBOUNDED)
        self.assertEqual(params.lower_bound, 0.0)
        self.assertEqual(params.upper_bound, 0.0)


class TestMetalogParameters(unittest.TestCase):
    """Tests for MetalogParameters class."""

    def test_instantiation(self):
        """Test MetalogParameters can be instantiated."""
        from metalog_jax.base.enums import MetalogBoundedness, MetalogFitMethod
        from metalog_jax.base.parameters import MetalogParameters

        params = MetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
            method=MetalogFitMethod.OLS,
            num_terms=5,
        )

        self.assertEqual(params.method, MetalogFitMethod.OLS)
        self.assertEqual(params.num_terms, 5)

    def test_inherits_from_base(self):
        """Test MetalogParameters inherits from MetalogParametersBase."""
        from metalog_jax.base.parameters import MetalogParameters, MetalogParametersBase

        self.assertTrue(issubclass(MetalogParameters, MetalogParametersBase))


class TestSPTMetalogParameters(unittest.TestCase):
    """Tests for SPTMetalogParameters class."""

    def test_instantiation(self):
        """Test SPTMetalogParameters can be instantiated."""
        from metalog_jax.base.enums import MetalogBoundedness
        from metalog_jax.base.parameters import SPTMetalogParameters

        params = SPTMetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
            alpha=0.1,
        )

        self.assertEqual(params.alpha, 0.1)

    def test_inherits_from_base(self):
        """Test SPTMetalogParameters inherits from MetalogParametersBase."""
        from metalog_jax.base.parameters import (
            MetalogParametersBase,
            SPTMetalogParameters,
        )

        self.assertTrue(issubclass(SPTMetalogParameters, MetalogParametersBase))


class TestMetalogRandomVariableParameters(unittest.TestCase):
    """Tests for MetalogRandomVariableParameters class."""

    def test_instantiation_with_jax_prng(self):
        """Test instantiation with JAX PRNG parameters."""
        from metalog_jax.base.parameters import MetalogRandomVariableParameters
        from metalog_jax.utils import JaxUniformDistributionParameters

        prng_params = JaxUniformDistributionParameters(seed=42)
        params = MetalogRandomVariableParameters(
            prng_params=prng_params,
            size=100,
        )

        self.assertEqual(params.size, 100)

    def test_validates_positive_size(self):
        """Test that size must be positive."""
        from metalog_jax.base.parameters import MetalogRandomVariableParameters
        from metalog_jax.utils import JaxUniformDistributionParameters

        prng_params = JaxUniformDistributionParameters(seed=42)

        with self.assertRaises(AssertionError):
            MetalogRandomVariableParameters(
                prng_params=prng_params,
                size=-1,
            )


if __name__ == "__main__":
    unittest.main()
