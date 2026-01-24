# Copyright: Travis Jefferies 2026
"""Tests for metalog_jax.base.data module."""

import unittest

import jax.numpy as jnp


class TestDataImport(unittest.TestCase):
    """Test that data classes can be imported from the new module location."""

    def test_import_metalog_base_data(self):
        """Test MetalogBaseData can be imported."""
        from metalog_jax.base.data import MetalogBaseData

        self.assertTrue(callable(MetalogBaseData))

    def test_import_metalog_input_data(self):
        """Test MetalogInputData can be imported."""
        from metalog_jax.base.data import MetalogInputData

        self.assertTrue(callable(MetalogInputData))


class TestMetalogBaseData(unittest.TestCase):
    """Tests for MetalogBaseData class."""

    def test_instantiation(self):
        """Test MetalogBaseData can be instantiated directly."""
        from metalog_jax.base.data import MetalogBaseData

        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([0.1, 0.5, 0.9])
        data = MetalogBaseData(x=x, y=y, precomputed_quantiles=True)

        self.assertEqual(len(data.x), 3)
        self.assertEqual(len(data.y), 3)
        self.assertTrue(data.precomputed_quantiles)

    def test_immutability(self):
        """Test MetalogBaseData is immutable."""
        from metalog_jax.base.data import MetalogBaseData

        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([0.1, 0.5, 0.9])
        data = MetalogBaseData(x=x, y=y, precomputed_quantiles=True)

        with self.assertRaises((AttributeError, TypeError)):
            data.x = jnp.array([4.0, 5.0, 6.0])


class TestMetalogInputData(unittest.TestCase):
    """Tests for MetalogInputData class."""

    def test_from_values_with_precomputed_quantiles(self):
        """Test from_values factory with precomputed quantiles."""
        from metalog_jax.base.data import MetalogInputData

        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([0.1, 0.5, 0.9])
        data = MetalogInputData.from_values(x=x, y=y, precomputed_quantiles=True)

        self.assertEqual(len(data.x), 3)
        self.assertEqual(len(data.y), 3)

    def test_from_values_with_raw_samples(self):
        """Test from_values factory with raw sample data."""
        from metalog_jax.base.data import MetalogInputData

        samples = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = jnp.array([0.1, 0.5, 0.9])
        data = MetalogInputData.from_values(x=samples, y=y, precomputed_quantiles=False)

        # After transformation, x should have same length as y
        self.assertEqual(len(data.x), 3)
        self.assertEqual(len(data.y), 3)

    def test_from_values_validates_probability_range(self):
        """Test from_values validates probability range."""
        from metalog_jax.base.data import MetalogInputData

        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([0.0, 0.5, 1.0])  # Invalid: includes 0 and 1

        with self.assertRaises(ValueError):
            MetalogInputData.from_values(x=x, y=y, precomputed_quantiles=True)

    def test_from_values_validates_ascending_order(self):
        """Test from_values validates strictly ascending order."""
        from metalog_jax.base.data import MetalogInputData

        x = jnp.array([3.0, 2.0, 1.0])  # Descending - invalid
        y = jnp.array([0.1, 0.5, 0.9])

        with self.assertRaises(AssertionError):
            MetalogInputData.from_values(x=x, y=y, precomputed_quantiles=True)


if __name__ == "__main__":
    unittest.main()
