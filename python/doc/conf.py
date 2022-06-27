
# -- Path setup --------------------------------------------------------------

import os
import sys
import re
import numpy
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'biPCA'
copyright = '2021, Jay S. Stanley III, Thomas Zhang, Boris Landa, Yuval Kluger'
author = 'Jay S. Stanley III, Thomas Zhang, Boris Landa, Yuval Kluger'

# The full version, including alpha/beta/rc tags
import bipca
version = re.sub(r'(\d+\.\d+)\.\d+(.*)', r'\1\2', bipca.__version__)
version = re.sub(r'(\.dev\d+).*?$', r'\1', version)
release = bipca.__version__
today_fmt = '%B %d, %Y'
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.graphviz',
    'sphinx.ext.ifconfig',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx.ext.imgmath',
]
master_doc='index'
autodoc_typehints = "none"


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
html_sidebars = {
    "**": ["search-field", "sidebar-nav-bs"]
}
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
add_function_parentheses = False

numpydoc_attributes_as_param_list = False
napoleon_preprocess_types = True
##external links:
extlinks = {'log': ('https://github.com/scottgigante/tasklogger/#tasklogger%s',
                      '')}
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_additional_pages = {
    'index': 'indexcontent.html',
}

html_title = "%s v%s Manual" % (project, version)
html_static_path = ['_static']
html_last_updated_fmt = '%b %d, %Y'

autosummary_generate = True
autosummary_imported_members = True
numpydoc_show_class_members = True 

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable',None),
    'neps': ('https://numpy.org/neps', None),
    'python': ('https://docs.python.org/dev', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'imageio': ('https://imageio.readthedocs.io/en/stable', None),
    'skimage': ('https://scikit-image.org/docs/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'scipy-lecture-notes': ('https://scipy-lectures.org', None),
}


html_use_modindex = True
html_copy_source = False
html_domain_indices = False
html_file_suffix = '.html'

default_role = "autolink"


import inspect
from os.path import relpath, dirname

for name in ['sphinx.ext.linkcode', 'numpydoc.linkcode']:
    try:
        __import__(name)
        extensions.append(name)
        break
    except ImportError:
        pass
else:
    print("NOTE: linkcode extension not found -- no links to source generated")


def _get_c_source_file(obj):
    if issubclass(obj, numpy.generic):
        return r"core/src/multiarray/scalartypes.c.src"
    elif obj is numpy.ndarray:
        return r"core/src/multiarray/arrayobject.c"
    else:
        # todo: come up with a better way to generate these
        return None


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # strip decorators, which would resolve to the source of the decorator
    # possibly an upstream bug in getsourcefile, bpo-1764286
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)

    fn = None
    lineno = None

    # Make a poor effort at linking C extension types
    if isinstance(obj, type) and obj.__module__ == 'numpy':
        fn = _get_c_source_file(obj)

    if fn is None:
        try:
            fn = inspect.getsourcefile(obj)
        except Exception:
            fn = None
        if not fn:
            return None

        try:
            source, lineno = inspect.getsourcelines(obj)
        except Exception:
            lineno = None

        fn = relpath(fn, start=dirname(numpy.__file__))

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    if 'dev' in numpy.__version__:
        return "https://github.com/numpy/numpy/blob/main/numpy/%s%s" % (
           fn, linespec)
    else:
        return "https://github.com/numpy/numpy/blob/v%s/numpy/%s%s" % (
           numpy.__version__, fn, linespec)

from pygments.lexers import CLexer
from pygments.lexer import inherit, bygroups
from pygments.token import Comment

class NumPyLexer(CLexer):
    name = 'NUMPYLEXER'

    tokens = {
        'statements': [
            (r'@[a-zA-Z_]*@', Comment.Preproc, 'macro'),
            inherit,
        ],
    }