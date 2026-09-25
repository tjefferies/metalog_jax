# Copyright: Travis Jefferies 2026
"""Tests for metalog_jax.base.enums module."""

import unittest
from enum import IntEnum


class TestEnumsImport(unittest.TestCase):
    """Test that enums can be imported from the new module location."""

    def test_import_custom_int_enum(self):
        """Test CustomIntEnum can be imported."""
        from metalog_jax.base.enums import CustomIntEnum

        self.assertTrue(issubclass(CustomIntEnum, IntEnum))

    def test_import_metalog_boundedness(self):
        """Test MetalogBoundedness can be imported."""
        from metalog_jax.base.enums import MetalogBoundedness

        self.assertTrue(issubclass(MetalogBoundedness, IntEnum))

    def test_import_metalog_fit_method(self):
        """Test MetalogFitMethod can be imported."""
        from metalog_jax.base.enums import MetalogFitMethod

        self.assertTrue(issubclass(MetalogFitMethod, IntEnum))

    def test_import_metalog_plot_options(self):
        """Test MetalogPlotOptions can be imported."""
        from metalog_jax.base.enums import MetalogPlotOptions

        self.assertTrue(issubclass(MetalogPlotOptions, IntEnum))


class TestCustomIntEnum(unittest.TestCase):
    """Tests for CustomIntEnum base class."""

    def test_from_value_returns_correct_member(self):
        """Test from_value returns the correct enum member."""
        from metalog_jax.base.enums import MetalogBoundedness

        result = MetalogBoundedness.from_value(1)
        self.assertEqual(result, MetalogBoundedness.UNBOUNDED)

    def test_from_value_raises_on_invalid_value(self):
        """Test from_value raises ValueError for invalid values."""
        from metalog_jax.base.enums import MetalogBoundedness

        with self.assertRaises(ValueError) as ctx:
            MetalogBoundedness.from_value(99)
        self.assertIn("99", str(ctx.exception))


class TestMetalogBoundedness(unittest.TestCase):
    """Tests for MetalogBoundedness enum."""

    def test_enum_members_exist(self):
        """Test all expected enum members exist."""
        from metalog_jax.base.enums import MetalogBoundedness

        self.assertTrue(hasattr(MetalogBoundedness, "UNBOUNDED"))
        self.assertTrue(hasattr(MetalogBoundedness, "STRICTLY_LOWER_BOUND"))
        self.assertTrue(hasattr(MetalogBoundedness, "STRICTLY_UPPER_BOUND"))
        self.assertTrue(hasattr(MetalogBoundedness, "BOUNDED"))

    def test_enum_member_count(self):
        """Test the enum has exactly 4 members."""
        from metalog_jax.base.enums import MetalogBoundedness

        self.assertEqual(len(MetalogBoundedness), 4)

    def test_enum_values_are_distinct(self):
        """Test all enum values are distinct integers."""
        from metalog_jax.base.enums import MetalogBoundedness

        values = [m.value for m in MetalogBoundedness]
        self.assertEqual(len(values), len(set(values)))

    def test_from_value_roundtrip(self):
        """Test from_value works for all members."""
        from metalog_jax.base.enums import MetalogBoundedness

        for member in MetalogBoundedness:
            result = MetalogBoundedness.from_value(member.value)
            self.assertEqual(result, member)


class TestMetalogFitMethod(unittest.TestCase):
    """Tests for MetalogFitMethod enum."""

    def test_enum_members_exist(self):
        """Test all expected enum members exist."""
        from metalog_jax.base.enums import MetalogFitMethod

        self.assertTrue(hasattr(MetalogFitMethod, "OLS"))
        self.assertTrue(hasattr(MetalogFitMethod, "Lasso"))
        self.assertTrue(hasattr(MetalogFitMethod, "Feasible"))

    def test_enum_member_count(self):
        """Test the enum has exactly 3 members."""
        from metalog_jax.base.enums import MetalogFitMethod

        self.assertEqual(len(MetalogFitMethod), 3)

    def test_from_value_roundtrip(self):
        """Test from_value works for all members."""
        from metalog_jax.base.enums import MetalogFitMethod

        for member in MetalogFitMethod:
            result = MetalogFitMethod.from_value(member.value)
            self.assertEqual(result, member)


class TestMetalogPlotOptions(unittest.TestCase):
    """Tests for MetalogPlotOptions enum."""

    def test_enum_members_exist(self):
        """Test all expected enum members exist."""
        from metalog_jax.base.enums import MetalogPlotOptions

        self.assertTrue(hasattr(MetalogPlotOptions, "PDF"))
        self.assertTrue(hasattr(MetalogPlotOptions, "CDF"))
        self.assertTrue(hasattr(MetalogPlotOptions, "SF"))

    def test_enum_member_count(self):
        """Test the enum has exactly 3 members."""
        from metalog_jax.base.enums import MetalogPlotOptions

        self.assertEqual(len(MetalogPlotOptions), 3)

    def test_is_int_enum(self):
        """Test MetalogPlotOptions is an IntEnum."""
        from metalog_jax.base.enums import MetalogPlotOptions

        self.assertTrue(issubclass(MetalogPlotOptions, IntEnum))
        self.assertIsInstance(MetalogPlotOptions.PDF.value, int)


if __name__ == "__main__":
    unittest.main()
