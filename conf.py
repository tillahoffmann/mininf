project = "minivb"
author = "Till Hoffmann"
copyright = "since 2023"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]
master_doc = "README"
exclude_patterns = [
    ".pytest_cache",
]
autodoc_typehints_format = "short"
myst_enable_extensions = [
    "dollarmath",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
