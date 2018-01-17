Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. note::
    This section is only useful for developers who need to update the documentation.

We use:

- `Sphinx <http://sphinx-doc.org/>` for generating the documentation
- `Doxygen <http://www.stack.nl/~dimitri/doxygen/>` to extract information from the source code
- `Breathe <https://breathe.readthedocs.org/en/latest/>`_ to link Sphinx and doxygen
- `sphinx-versioning <https://robpol86.github.io/sphinxcontrib-versioning/>`_ (custom version) to generate a documentation for every version / branch

Install sphinx via pip: ::

    sudo pip install Sphinx
    sudo pip install sphinxcontrib-bibtex
    sudo pip install breathe
    sudo pip install git+https://github.com/resibots/sphinxcontrib-versioning.git@resibots

.. warning::

  On Mac OSX, do not use `brew install sphinx` because this is not the right sphinx

.. note::
    For Python 3, use `pip3` instead of `pip`

Install the Resibots theme for Sphinx::

    git clone https://github.com/resibots/sphinx_resibots_theme
    export SPHINX_RESIBOTS_THEME="/home/me/path/to/sphinx_resibots_theme"

Install `doxygen <http://www.stack.nl/~dimitri/doxygen/>`_ via your package manager (e.g. apt-get / brew)::

    apt-get install doxygen

In the main limbo directory::

    ./waf docs

About sphinx and ReStructuredText:
  - `There is a tutorial <http://sphinx-doc.org/tutorial.html>`_,
  - `Primer for ReStructuredText <http://sphinx-doc.org/rest.html>`_, the markup language of Sphinx,
  - `markup specific to Sphinx <http://sphinx-doc.org/markup/index.html>`_,
  - `About C++ in Sphinx <http://sphinx-doc.org/domains.html#id2>`_
  - `Breathe (bridge between sphinx and doxygen) <https://breathe.readthedocs.org/en/latest/>`_
  - `Sphinx-versioning  <https://robpol86.github.io/sphinxcontrib-versioning/>`_

