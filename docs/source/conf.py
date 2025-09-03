# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# The path below assumes your package is in `src/cgr_smiles` relative to the project root.
# From `docs/conf.py`, to reach `src`, you go up two levels (`../../`).
sys.path.insert(0, os.path.abspath("../../src"))

# Get version directly from your package
try:
    from cgr_smiles import __version__
except ImportError:
    __version__ = "0.0.0+unknown"  # Fallback if package not installed or in path

project = "cgr_smiles"
copyright = "2025, Charlotte Gerhaher"
author = "Charlotte Gerhaher"
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Automatically pulls docstrings from code
    "sphinx.ext.napoleon",  # Understands Google/NumPy style docstrings
    "sphinx_autodoc_typehints",  # Nicely formats type hints
    "sphinx.ext.viewcode",  # Links to source code for documented objects
    "sphinx.ext.todo",  # For tracking TODOs in docs
    "sphinx.ext.autosummary",  # Generates summary tables (useful for API)
    "sphinx.ext.intersphinx",  # Link to other projects' docs (e.g., pandas, rdkit)
    "sphinx_togglebutton",  # For interactive elements
    "nbsphinx",  # For integrating Jupyter notebooks
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]


napoleon_google_docstring = True
napoleon_numpy_docstring = False  # Keep False if you're not using NumPy style
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_rtype = True  # Recommended for Google style docstrings


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "rdkit": ("https://www.rdkit.org/docs/", None),
    # You can add more like:
    # 'numpy': ('https://numpy.org/doc/stable/', None),
    # 'tqdm': ('https://tqdm.github.io/docs/', None),
    # 'rich': ('https://rich.readthedocs.io/en/stable/', None),
}
