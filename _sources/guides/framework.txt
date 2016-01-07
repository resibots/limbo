Using Limbo as an environment for experiments
=============================================

The typical use case of Limbo for research in Bayesian Optimization is:

- we design an experiment that uses some components of Limbo
- we want to compare if variant X of the experiment (e.g. with kernel XXX) is better than variant Y (e.g. with kernel YYY)
- because the algorithms that we use have some stochastic components (initialization, inner optimization, ...), we usually need to replicate each experiment (typically, we use 30 replicates) in order to do some statistics (see  `Matplotlib for Papers <http://www.github.com/jbmouret/matplotlib_for_papers>`_ for a tutorial about how to draw nice box plots with these statistics).

Limbo provides basics tools to make these steps easier. They are mostly additions to ``waf`` (see our :ref:`FAQ about waf <faq-waf>`). For users who are used to ROS, you can see these additions as our 'catkin for Bayesian optimization'.

How to add / compile your experiment?
-------------------------------------

- add a directory called ``exp`` at the root the limbo tree
- add a directory for your experiment (e.g. ``my_experiment``)
- add a ``wscript`` in this directory:

.. code-block:: python

  #!/usr/bin/env python
  # encoding: utf-8

  import limbo
  import commands

  def options(opt):
    # you add command line options here
    pass


  def configure(conf):
      # you can add configurations here
      # for instance, to link with ncurses:
      conf.env['LIB_NCURSE'] += ['ncurses']

  def build(bld):
      obj = bld.program(features = 'cxx',
                        source = 'my_file1.cpp my_file2.cpp',# separate with spaces
                        includes = '. ../../src/',
                        target = 'belty',
                        uselib =  'BOOST EIGEN TBB NCURSE', # add NCURSE here to actually link with it
                        use = 'limbo')

- compile with: ``./waf --exp my_experiment``
- if you added configure options, you need to do a ``./waf configure --exp my_experiment`` first


How to submit jobs with limbo on clusters?
------------------------------------------

OAR (``oarsub``) and Torque (``qsub``) are supported. The system is very similar to the system used in `Sferes2 <http://github.com/sferes2/sferes2>`_, therefore if you know Sferes2, it will be easy for you.

Depending on the scheduler, we have to commands:

- ``./waf --qsub=your_json_file.json``
- ``./waf --oar=your_json_file.json``

The json file should look like this (for both OAR or Torque):

.. code-block:: javascript

    {
     "exps" : ["hexa_duty_text"],
     "bin_dir" : "/nfs/hal01/jmouret/git/sferes2/build/default/exp/hexa_duty_cycle",
     "res_dir" : "/nfs/hal01/jmouret/data/maps_hexapod-slippy/",
     "email" : "JBM",
     "wall_time" : "270:00:00",
     "nb_runs" : 2,
     "nb_cores" : 24,
    }

Explanations:

- ``exps`` is the list of the experiments; these are binary names that will be found in ``bin_dir``; this is an array: you can have as many binary names as you want (separate with a comma)
- ``bin_dir`` is the directory that contains the binaries that correspond to the experiments; be careful that the directory needs to be reachable from all the nodes (typically, it should be on NFS)
- ``res_dir`` is where to store the results. Limbo will create a directory for each experiments. For instance, here is the directory structure for this json:


::

  data/
  +-- hexa_duty_text/
    +-- exp_0/
    +-- exp_1/


- ``email`` could be your e-mail (to receive an e-mail when the job is finished). It is currently not supported for OAR;
- ``wall_time`` is the number of hours for each replicate of each experiment (be careful that your job will be killed at the end of this time; however, if you put a number to high, your job will be redirected to low-priority queues)
- ``nb_runs`` is the number of replicates of each experiment;
- ``nb_cores`` is the number of cores for a single experiment (MPI is currently not supported in limbo).

Variants
--------
A very common use case is to compare variant XX to variant YY of an algorithm. Usually, only a few lines of code are different (like, calling kernel X or kernel Y). Limbo is designed to create a binary for each variant by defining a few constant at the beginning of a source file.

For instance, let's say we have a file called ``multi.cpp`` for which we want to compare two algorithms, ``Parego`` and ``EHVI``:

.. code-block:: cpp

  //.... code
  #ifdef PAREGO
    Parego<Params, stat_fun<stat_t>> opt;
  #else
    Ehvi<Params, stat_fun<stat_t>> opt;
  #endif
  // ...

We can create two variants in the ``wscript``, as follows:

.. code-block:: python

  #! /usr/bin/env python
  import limbo
  def build(bld):

    limbo.create_variants(bld,
                        source = 'multi.cpp',
                        uselib_local = 'limbo',
                        uselib = 'BOOST EIGEN TBB SFERES',
                        variants = ['PAREGO',
                                    'EHVI'])


Limbo will create two files:

- ``multi_parego.cpp``, which is the ``multi.cpp`` file with a ``#define PAREGO`` at the first line
- ``multi_ehvi.cpp``, which is the ``multi.cpp`` file with a ``#define EHVI`` at the first line

**You should never edit these files**: they will be re-generated each time you will compile.

You can add as many defines as you like (or even generate them with python code), for instance:


.. code-block:: python

  #! /usr/bin/env python
  import limbo
  def build(bld):

    limbo.create_variants(bld,
                        source = 'multi.cpp',
                        uselib_local = 'limbo',
                        uselib = 'BOOST EIGEN TBB SFERES',
                        variants = ['PAREGO MOP2 DIM2',
                                    'EHVI ZDT2 DIM6'])


This will create ``multi_parego_mop2_dim2`` and ``multi_ehvi_zdt2_dim6``.

Using ``./waf --exp your_experiment`` will compile all the corresponding libraries. If you want to compile a single variant, you can use the ``--target`` option: ``./waf --exp your_experiment --target parego_mop2_dim2`.

If you have more than one file, you will need to first compile a static library, then link with it in the variant.
