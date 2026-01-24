"""Sphinx configuration for metalog-jax documentation."""

import os
import sys

# Add the project root to the path for autodoc
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "metalog-jax"
copyright = "2026, Travis Jefferies"
author = "Travis Jefferies"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "nbsphinx",
]

# nbsphinx settings
nbsphinx_execute = "never"  # Use pre-executed notebooks with stored outputs
nbsphinx_allow_errors = False


templates_path = ["_templates"]
exclude_patterns = []

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# sphinx-autodoc-typehints settings
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True

# Intersphinx mapping for external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_material"

html_theme_options = {
    "nav_title": "metalog-jax",
    "color_primary": "indigo",
    "color_accent": "pink",
    "repo_url": "https://github.com/tjefferies/metalog_jax",
    "repo_name": "metalog_jax",
    "repo_type": "github",
    "globaltoc_depth": 3,
    "globaltoc_collapse": True,
    "globaltoc_includehidden": True,
    "heroes": {
        "index": "JAX implementation of the Metalog distribution",
    },
}

html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

html_static_path = ["_static"]
html_title = "metalog-jax"

# Custom CSS file for styling fixes
html_css_files = [
    "custom.css",
]

# Create _static directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), "_static"), exist_ok=True)
