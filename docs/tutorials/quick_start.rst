Quick Start
=========================================================

Get limbo
------------

To get **limbo**, simply clone the source code from https://github.com/resibots/limbo with git, or download it
as a zip.

Install the dependencies:
----------------------------

For Ubuntu:

::

  apt-get install libeigen3-dev libboost* libtbb*

For OSX with brew:

::

  brew install eigen3
  brew install boost

We highly recommend that you install NLOpt. Infortunately, the Ubuntu packages are missing the C++. You can get NLOpt here: http://ab-initio.mit.edu/wiki/index.php/NLopt

::

  wget http://ab-initio.mit.edu/wiki/index.php/NLopt/TODO
  tar zxvf TODO
  cd TODO
   ./configure --with-cxx --enable-shared --without-python --without-matlab --without-octave
   sudo make install


For more options and troubleshootings, see the :ref:`Compilation tutorial <compilation-tutorial>`.

Compile Limbo
-----------------

::

  ./waf configure
  ./waf build

Create a new experiment
---------------------------

::

  ./waf --create test

See the :ref:`Framework guide <framework-guide>`

Edit the "Eval" function to define the function that you want to optimized
-------------------------------------------------------------------------

::

  $EDITOR exp/test.cpp

Build your experiment
-----------------------

::

  ./waf --exp test

Run your experiment
-----------------------
::

  build/exp/test

Analyze the results
--------------------
