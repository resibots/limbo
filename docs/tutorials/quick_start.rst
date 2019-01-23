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

  apt-get install libeigen3-dev libboost-serialization-dev libboost-filesystem-dev libboost-test-dev libboost-program-options-dev libboost-thread-dev libtbb-dev

For OSX with brew:

::

  brew install eigen3
  brew install boost

We highly recommend that you install NLOpt. Unfortunately, the Ubuntu packages do not provide NLOpt's C++ bindings. You can get NLOpt here: http://ab-initio.mit.edu/wiki/index.php/NLopt [mirror: http://members.loria.fr/JBMouret/mirrors/nlopt-2.4.2.tar.gz]

For Ubuntu / Debian:
::

  sudo apt-get -qq update
  sudo apt-get -qq --yes --force-yes install autoconf automake
  wget http://members.loria.fr/JBMouret/mirrors/nlopt-2.4.2.tar.gz
  tar -zxvf nlopt-2.4.2.tar.gz && cd nlopt-2.4.2
  ./configure -with-cxx --enable-shared --without-python --without-matlab --without-octave
  sudo make install
  sudo ldconfig
  cd ..

For OSX:
::

  wget http://members.loria.fr/JBMouret/mirrors/nlopt-2.4.2.tar.gz
  tar -zxvf nlopt-2.4.2.tar.gz && cd nlopt-2.4.2
  ./configure --with-cxx --enable-shared --without-python --without-matlab --without-octave
  sudo make install
  cd ..


For more options and troubleshooting, see the :ref:`Compilation tutorial <compilation-tutorial>`.

Compile Limbo
-----------------

To get **limbo**, simply clone the source code from https://github.com/resibots/limbo with git, or download it
as a zip. To compile it:
   
::

  ./waf configure
  ./waf build

For more options and troubleshooting, see the :ref:`Compilation tutorial <compilation-tutorial>`.


Create a new experiment
---------------------------

::

  ./waf --create test

For more information about experiments in Limbo, see the :ref:`Framework guide <framework-guide>`

Edit the "Eval" function to define the function that you want to optimized
-------------------------------------------------------------------------

::

  $EDITOR exp/test/test.cpp

The part to edit is between line 56 and line 63:

.. code-block:: c++

  Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
  {
      double y = 0;
    // YOUR CODE HERE
    // ...
    // return a 1-dimensional vector
    return tools::make_vector(y);
  }

Limbo always maximizes and by default in :math:`[0,1]`. For more information, see the :ref:`Basic example <basic-example>`.


Build your experiment
-----------------------

::

  ./waf configure --exp test
  ./waf --exp test

Run your experiment
-----------------------
::

  build/exp/test/test

Analyze the results
--------------------

The results are in yourcomputer-date-hour-pid. For instance: ``wallepro-perso.loria.fr_2016-09-15_19_43_50_74198``.
