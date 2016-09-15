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

We highly recommend that you install NLOpt. Infortunately, the Ubuntu packages are missing the C++. You can get NLOpt here: http://ab-initio.mit.edu/wiki/index.php/NLopt [mirror: http://members.loria.fr/JBMouret/mirrors/nlopt-2.4.2.tar.gz]

For Ubuntu / Debian:
::

  sudo apt-get -qq update
  sudo apt-get -qq --yes --force-yes install autoconf automake
  wget http://members.loria.fr/JBMouret/mirrors/nlopt-2.4.2.tar.gz
  tar -zxvf nlopt-2.4.2.tar.gz && cd nlopt-2.4.2
  ./configure -with-cxx --enable-shared --without-python --without-matlab --without-octave
  sudo make install
  sudo ldconfig

For OSX:
::

  wget http://members.loria.fr/JBMouret/mirrors/nlopt-2.4.2.tar.gz
  tar -zxvf nlopt-2.4.2.tar.gz && cd nlopt-2.4.2
  ./configure -with-cxx --enable-shared --without-python --without-matlab --without-octave
  sudo make install


For more options and troubleshootings, see the :ref:`Compilation tutorial <compilation-tutorial>`.

Compile Limbo
-----------------

::

  ./waf configure
  ./waf build

For more options and troubleshootings, see the :ref:`Compilation tutorial <compilation-tutorial>`.


Create a new experiment
---------------------------

::

  ./waf --create test

See the :ref:`Framework guide <framework-guide>`

Edit the "Eval" function to define the function that you want to optimized
-------------------------------------------------------------------------

::

  $EDITOR exp/test/test.cpp


For more information, see the :ref:`Basic example <basic-example>`


Build your experiment
-----------------------

::

  ./waf --exp test

Run your experiment
-----------------------
::

  build/exp/test/test

Analyze the results
--------------------

The results are in
