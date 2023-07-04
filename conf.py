import os


project = "mininf"
author = "Till Hoffmann"
copyright = "since 2023"
html_theme = "pydata_sphinx_theme"
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]
master_doc = "README"
exclude_patterns = [
    ".pytest_cache",
    "**/.ipynb_checkpoints",
    "**/.jupyter_cache",
    "**/jupyter_execute",
    "playground",
]
nitpick_ignore = [
    ("py:class", "torch.Size"),
]
add_module_names = False
autodoc_typehints_format = "short"
myst_enable_extensions = [
    "dollarmath",
]
nb_execution_mode = "off" if "NOEXEC" in os.environ else "cache"
nb_execution_timeout = 60
nb_execution_allow_errors = False
nb_execution_raise_on_error = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
