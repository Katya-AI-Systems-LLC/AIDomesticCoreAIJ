# Configuration file for the Sphinx documentation builder.
# Full list of options available: https://www.sphinx-doc.org/en/master/config

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'AI Platform'
copyright = '2024, AI Platform Team'
author = 'AI Platform Team'
release = '1.0.0'
version = '1.0'

# General configuration
extensions = [
    # Built-in Sphinx extensions
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.imgmath',
    
    # Third-party extensions
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinxcontrib.openapi',
    'sphinxcontrib.autodoc_pydantic',
    'myst_parser',
    'nbsphinx',
    'sphinx_design',
    'sphinx_tabs.tabs',
]

# Add napoleon extension for Google-style docstrings
extensions.append('sphinx.ext.napoleon')

# Napoleon configuration
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'show-inheritance': True,
}

# Autosummary configuration
autosummary_generate = True

# Source file suffix and formats
source_suffix = {
    '.rst': None,
    '.md': 'myst-nb',
}

# Root document
root_doc = 'index'

# Exclude patterns
exclude_patterns = [
    '_build',
    '**.ipynb_checkpoints',
    '.DS_Store',
]

# Language
language = 'en'

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': 'view',
    'style_nav_header_background': '#1f1c1c',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'titles_only': False,
    'github_url': 'https://github.com/yourusername/aiplatform',
}

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
}

# Mathjax configuration for math rendering
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

# OpenAPI/Swagger configuration
openapi_root_dir = 'reference/api'
openapi_extra_static_dir = '_static/openapi'

# Copy button configuration
copybutton_copy_empty_lines = False
copybutton_selector = "div:not(.no-copy)>div.highlight>pre"

# nbsphinx configuration (Jupyter notebook support)
nbsphinx_kernel_name = 'python3'
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_assume_equations = True

# Pydantic configuration
autodoc_pydantic_model_show_json_schema_extra = False
autodoc_pydantic_settings_show_json_schema_extra = False

# HTML output
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'

html_context = {
    'display_github': True,
    'github_user': 'aiplatform-team',
    'github_repo': 'aiplatform',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

# LaTeX output options
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{listings}
\usepackage{xcolor}
\lstset{
    basicstyle=\ttfamily,
    breaklines=true,
    language=Python,
    keywordstyle=\color{blue},
    commentstyle=\color{gray},
    stringstyle=\color{red},
}
''',
}

latex_documents = [
    ('index', 'aiplatform.tex', 'AI Platform Documentation', 'AI Platform Team', 'manual'),
]

# EPUB output
epub_title = 'AI Platform Documentation'
epub_author = 'AI Platform Team'
epub_publisher = 'AI Platform'
epub_copyright = '2024'

# Man pages
man_pages = [
    ('cli/index', 'aiplatform', 'AI Platform CLI', ['AI Platform Team'], 1),
]

# Texinfo
texinfo_documents = [
    ('index', 'aiplatform', 'AI Platform Documentation', 'AI Platform Team', 'aiplatform',
     'Comprehensive AI and quantum computing platform.', 'Miscellaneous'),
]

# Linkcheck configuration
linkcheck_ignore = [
    r'https://github.com/[^/]+/[^/]+/issues',
    r'https://github.com/[^/]+/[^/]+/pulls',
]

# Suppress certain warnings
suppress_warnings = [
    'app.add_config_value',
]

# HTML 5 output
html_experimental_html5_writer = True

# Setup function for custom configurations
def setup(app):
    # Add custom CSS
    app.add_css_file('custom.css')
    
    # Add custom JS
    app.add_js_file('custom.js')
    
    # Connect signals
    app.connect('config-inited', on_config_inited)
    app.connect('env-before-read-docs', on_env_before_read_docs)
    
    return {
        'version': '1.0.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }

def on_config_inited(app, config):
    """Called when the config object has been initialized."""
    pass

def on_env_before_read_docs(app, env, docnames):
    """Called before reading files."""
    pass

# Maximum inline math length
math_number_all = False
math_eqref_format = 'Equation {number}'

# Enable copy button on code blocks
copybutton_prompt_primary = '>>> '
copybutton_prompt_continuation = '... '
copybutton_only_copy_prompt_lines = True

# Warn about missing references (links to non-existent docs)
nitpicky = False

# Show line numbers for code blocks
highlight_language = 'python3'
pygments_style = 'sphinx'

# Custom CSS for better styling
html_css_files = [
    'custom.css',
]

# Custom JavaScript
html_js_files = [
    'custom.js',
]

# PDF configuration (if using rst2pdf)
try:
    import rst2pdf
    extensions.append('rst2pdf.pdfbuilder')
    pdf_documents = [
        ('index', 'aiplatform', 'AI Platform Documentation', 'AI Platform Team'),
    ]
    pdf_stylesheets = ['sphinx', 'kerning', 'a4']
except ImportError:
    pass

# Additional configuration values
# Timeout for external link checking (seconds)
linkcheck_timeout = 30

# Retry count for external link checking
linkcheck_retries = 2

# Custom roles and directives can be registered here
# Example: app.add_role('custom', custom_role)
