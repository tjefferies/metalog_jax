# Copyright: Travis Jefferies 2026
"""Tests for metalog_jax.base.core module."""

import unittest

import jax.numpy as jnp


class TestCoreImport(unittest.TestCase):
    """Test that core class can be imported from the new module location."""

    def test_import_metalog_base(self):
        """Test MetalogBase can be imported."""
        from metalog_jax.base.core import MetalogBase

        self.assertTrue(callable(MetalogBase))


class TestMetalogBase(unittest.TestCase):
    """Tests for MetalogBase class."""

    def test_instantiation(self):
        """Test MetalogBase can be instantiated."""
        from metalog_jax.base.core import MetalogBase
        from metalog_jax.base.enums import MetalogBoundedness, MetalogFitMethod
        from metalog_jax.base.parameters import MetalogParameters

        params = MetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
            method=MetalogFitMethod.OLS,
            num_terms=3,
        )
        a = jnp.array([1.0, 0.5, 0.1])

        metalog = MetalogBase(metalog_params=params, a=a)

        self.assertEqual(metalog.boundedness, MetalogBoundedness.UNBOUNDED)
        self.assertEqual(metalog.num_terms, 3)

    def test_properties(self):
        """Test MetalogBase properties work correctly."""
        from metalog_jax.base.core import MetalogBase
        from metalog_jax.base.enums import MetalogBoundedness, MetalogFitMethod
        from metalog_jax.base.parameters import MetalogParameters

        params = MetalogParameters(
            boundedness=MetalogBoundedness.BOUNDED,
            lower_bound=0.0,
            upper_bound=100.0,
            method=MetalogFitMethod.Lasso,
            num_terms=5,
        )
        a = jnp.array([1.0, 0.5, 0.1, 0.05, 0.02])

        metalog = MetalogBase(metalog_params=params, a=a)

        self.assertEqual(metalog.lower_bound, 0.0)
        self.assertEqual(metalog.upper_bound, 100.0)
        self.assertEqual(metalog.num_terms, 5)

    def test_static_transformations(self):
        """Test static transformation methods."""
        from metalog_jax.base.core import MetalogBase

        x = jnp.array([2.0, 3.0, 4.0])

        # Test strictly lower bound transform
        result = MetalogBase.strictly_lower_bound_quantile_transform(x, 1.0)
        expected = jnp.log(x - 1.0)
        self.assertTrue(jnp.allclose(result, expected))

        # Test strictly upper bound transform
        result = MetalogBase.strictly_upper_bound_quantile_transform(x, 5.0)
        expected = -jnp.log(5.0 - x)
        self.assertTrue(jnp.allclose(result, expected))

        # Test bounded transform
        result = MetalogBase.bounded_quantile_transform(x, 0.0, 5.0)
        expected = jnp.log((x - 0.0) / (5.0 - x))
        self.assertTrue(jnp.allclose(result, expected))

    def test_helper_terms(self):
        """Test helper term methods."""
        from metalog_jax.base.core import MetalogBase

        x = jnp.array([0.25, 0.5, 0.75])

        # Test delta term
        delta = MetalogBase.delta_term(x)
        self.assertTrue(jnp.allclose(delta, x - 0.5))

        # Test log term
        log_term = MetalogBase.log_term(x)
        self.assertTrue(jnp.allclose(log_term, jnp.log(x / (1 - x))))

        # Test product term
        product = MetalogBase.product_term(x)
        self.assertTrue(jnp.allclose(product, x * (1 - x)))


class TestMetalogBaseEquality(unittest.TestCase):
    """Tests for MetalogBase __eq__ method."""

    def _create_metalog_base(
        self,
        boundedness=None,
        lower_bound=0.0,
        upper_bound=0.0,
        method=None,
        num_terms=3,
        a=None,
    ):
        """Helper to create a MetalogBase instance."""
        from metalog_jax.base.core import MetalogBase
        from metalog_jax.base.enums import MetalogBoundedness, MetalogFitMethod
        from metalog_jax.base.parameters import MetalogParameters

        if boundedness is None:
            boundedness = MetalogBoundedness.UNBOUNDED
        if method is None:
            method = MetalogFitMethod.OLS
        if a is None:
            a = jnp.array([1.0, 0.5, 0.1])

        params = MetalogParameters(
            boundedness=boundedness,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            method=method,
            num_terms=num_terms,
        )
        return MetalogBase(metalog_params=params, a=a)

    def test_equal_metalog_instances(self):
        """Test that two identical MetalogBase instances are equal."""
        metalog1 = self._create_metalog_base()
        metalog2 = self._create_metalog_base()

        self.assertTrue(metalog1 == metalog2)

    def test_equal_metalog_different_a_values(self):
        """Test that metalogs with different coefficient arrays are not equal."""
        metalog1 = self._create_metalog_base(a=jnp.array([1.0, 0.5, 0.1]))
        metalog2 = self._create_metalog_base(a=jnp.array([1.0, 0.6, 0.1]))

        self.assertFalse(metalog1 == metalog2)

    def test_equal_metalog_different_boundedness(self):
        """Test that metalogs with different boundedness are not equal."""
        from metalog_jax.base.enums import MetalogBoundedness

        metalog1 = self._create_metalog_base(boundedness=MetalogBoundedness.UNBOUNDED)
        metalog2 = self._create_metalog_base(
            boundedness=MetalogBoundedness.STRICTLY_LOWER_BOUND
        )

        self.assertFalse(metalog1 == metalog2)

    def test_equal_metalog_different_method(self):
        """Test that metalogs with different methods are not equal."""
        from metalog_jax.base.enums import MetalogFitMethod

        metalog1 = self._create_metalog_base(method=MetalogFitMethod.OLS)
        metalog2 = self._create_metalog_base(method=MetalogFitMethod.Lasso)

        self.assertFalse(metalog1 == metalog2)

    def test_equal_metalog_different_num_terms(self):
        """Test that metalogs with different num_terms are not equal."""
        metalog1 = self._create_metalog_base(num_terms=3, a=jnp.array([1.0, 0.5, 0.1]))
        metalog2 = self._create_metalog_base(
            num_terms=5, a=jnp.array([1.0, 0.5, 0.1, 0.05, 0.02])
        )

        self.assertFalse(metalog1 == metalog2)

    def test_equal_metalog_different_bounds(self):
        """Test that metalogs with different bounds are not equal."""
        metalog1 = self._create_metalog_base(lower_bound=0.0, upper_bound=100.0)
        metalog2 = self._create_metalog_base(lower_bound=0.0, upper_bound=200.0)

        self.assertFalse(metalog1 == metalog2)

    def test_equal_with_non_metalog_returns_false(self):
        """Test that comparing MetalogBase with non-MetalogBase returns False."""
        metalog = self._create_metalog_base()

        self.assertFalse(metalog == "not a metalog")
        self.assertFalse(metalog == 42)
        self.assertFalse(metalog == None)

    def test_equal_with_subclass(self):
        """Test that MetalogBase can compare with its subclasses (Metalog)."""
        from metalog_jax.base.enums import MetalogBoundedness, MetalogFitMethod
        from metalog_jax.base.parameters import MetalogParameters
        from metalog_jax.metalog import Metalog

        params = MetalogParameters(
            boundedness=MetalogBoundedness.UNBOUNDED,
            lower_bound=0.0,
            upper_bound=0.0,
            method=MetalogFitMethod.OLS,
            num_terms=3,
        )
        a = jnp.array([1.0, 0.5, 0.1])

        metalog1 = Metalog(metalog_params=params, a=a)
        metalog2 = Metalog(metalog_params=params, a=a)

        self.assertTrue(metalog1 == metalog2)


if __name__ == "__main__":
    unittest.main()
