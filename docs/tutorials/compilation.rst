Installation, Compilation and Tests
=================================================

Installation
----------------------------

To install **limbo**, simply clone the source code from https://github.com/resibots/limbo with git, or download it
as a zip.

Dependencies
~~~~~~~~~~~~~

Required
+++++++++++++
* `Boost <http://www.boost.org>`_ , with the following libraries: serialization, filesystem, system, unit_test_framework, program_options, graph and thread
* `Eigen <http://eigen.tuxfamily.org>`_

Optional
+++++++++++++
* `Intel TBB <https://www.threadingbuildingblocks.org>`_ is not mandatory, but highly recommended
* `Intel MKL <https://software.intel.com/en-us/intel-mkl>`_ is supported as backend for Eigen. In our experience, it provided best results when compiling with Intel's Compiler
* `Sferes2 <https://github.com/sferes2/sferes2>`_ if you plan to use the multi-objective bayesian optimization algorithms

Compilation
----------------------------

We use  the `WAF <https://waf.io>`_  build system, which is provided with the **limbo** source code.
The first step is to configure your waf environment. For this, assuming that you are in the main limbo directory, you have to run the command: ::

    ./waf configure

Make sure that the waf file has execution rights.
