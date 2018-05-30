.. _framework-guide:

Using Limbo as an environment for scientific experiments
=========================================================

The typical use case of Limbo for research in Bayesian Optimization is:

- we design an experiment that uses some components of Limbo
- we want to know whether variant X of the experiment (e.g. with kernel XXX) is better than variant Y (e.g. with kernel YYY)
- because the algorithms that we use have some stochastic components (initialization, inner optimization, ...), we usually need to replicate each experiment (typically, we use 30 replicates) in order to do some statistics (see  `Matplotlib for Papers <http://www.github.com/jbmouret/matplotlib_for_papers>`_ for a tutorial about how to draw nice box plots with these statistics).

Limbo provides basics tools to make these steps easier. They are mostly additions to ``waf`` (see our :ref:`FAQ about waf <faq-waf>`). For users who are used to ROS, you can see these additions as our 'catkin for Bayesian optimization'.

**The use of these tools are optional**: you can use Limbo as header-only library in your own project.


What is a Limbo experiment?
-----------------------
Each time we want to investigate an idea (e.g. a particular function to optimize, a new kernel function, etc.), we create a new experiment in the directory ``exp``. For instance, we can have ``exp/test``. This directory should contain all the code that is specific to your experiment (.cpp files, but also .hpp, data files, etc.).

Experiments give you the following benefits:

- it keeps things organized (with code that is specific to a specific paper in a directory and generic code that is maintained by Limbo's team)
- Limbo provides a service to easily generate variants of an experiment (e.g. compare using kernel XX using kernel YY)
- experiments can be easily submitted to a cluster (``--oar=...``)
- experiments can be easily run multiple times locally (if you do not have access to a cluster), via ``--local`` or ``--loca-serial``


How to quickly create a new experiment?
----------------------------------------
To quickly create a new experiment, you can use ``./waf --create=your_name``. For instance ``./waf --create=test`` will create a new directory in exp/test with a ``wscript`` and a file called ``test.cpp``, based on a basic template.

The experiment can the be compiled using ``./waf --exp test``

If you want to customize the parameters, you can use the following options:

- ``--dim_in=DIM_IN``: Number of input dimensions for the function to optimize [default: 1]
- ``--dim_out=DIM_OUT``: Number of output dimensions for the function to optimize [default: 1]
- ``--bayes_opt_boptimizer_noise=BAYES_OPT_BOPTIMIZER_NOISE``: Acquisition noise of the function to optimize [default: 1e-6]
- ``--bayes_opt_bobase_stats_enabled``: Enable statistics [default: true]
- ``--init_randomsampling_samples=INIT_RANDOMSAMPLING_SAMPLES``: Number of samples used for the initialization [default: 10]
- ``--stop_maxiterations_iterations=STOP_MAXITERATIONS_ITERATIONS``: Number of iterations performed before stopping the optimization [default: 190


**These parameters can be changed later.** You will only need to open the generated cpp file and put the values you want.

How to add / compile your experiment?
-------------------------------------
If you do not want to use ``./waf --create``, you can do it yourself:

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

- compile with: ``./waf --exp my_experiment`` (from limbo's folder)
- if you added configure options, you need to do a ``./waf configure --exp my_experiment`` first


How to submit jobs with limbo on clusters?
------------------------------------------

OAR (``oarsub``) and Torque (``qsub``) are supported. The system is very similar to the system used in `Sferes2 <http://github.com/sferes2/sferes2>`_, therefore if you know Sferes2, it will be easy for you.

Depending on the scheduler, we have two commands:

- ``./waf --qsub=your_json_file.json``
- ``./waf --oar=your_json_file.json``

The json file should look like this (for both OAR or Torque):

.. code-block:: javascript

    [{
     "exps" : ["hexa_duty_text"],
     "bin_dir" : "/nfs/hal01/jmouret/git/sferes2/build/default/exp/hexa_duty_cycle",
     "res_dir" : "/nfs/hal01/jmouret/data/maps_hexapod-slippy/",
     "email" : "JBM",
     "wall_time" : "270:00:00",
     "nb_runs" : 2,
     "nb_cores" : 24
    },

    {
     "exps" : ["hexa_duty_graphic"],
     "bin_dir" : "/nfs/hal01/jmouret/git/sferes2/build/default/exp/hexa_duty_cycle",
     "res_dir" : "/nfs/hal01/jmouret/data/maps_hexapod-slippy-graphic/",
     "email" : "JBM",
     "wall_time" : "270:00:00",
     "nb_runs" : 2,
     "nb_cores" : 24
    }]

Explanations:

- ``exps`` is the list of the experiments; these are binary names that will be found in ``bin_dir``; this is an array: you can have as many binary names as you want (separated by a comma)
- ``bin_dir`` is the directory that contains the binaries that correspond to the experiments; be careful that the directory needs to be reachable from all the nodes (typically, it should be on NFS)
- ``res_dir`` is where to store the results. Limbo will create a directory for each experiments. For instance, here is the directory structure for this json::

    data/
    +-- hexa_duty_text/
      +-- exp_0/
      +-- exp_1/


- ``email`` could be your e-mail (to be notified when the job is finished). It is currently not supported for OAR;
- ``wall_time`` is the allocated number of hours for each replicate of each experiment. Be careful that your job will be killed at the end of this time; however, if you put a number to high, your job will be redirected to low-priority queues
- ``nb_runs`` is the number of replicates of each experiment;
- ``nb_cores`` is the number of cores for a single experiment (MPI is currently not supported in limbo).

Variants
--------
A very common use case is to compare variant XX to variant YY of an algorithm. Usually, only a few lines of code are different (like, calling kernel XXX or kernel YYY). Limbo is designed to create a binary for each variant by using defines (like defining constants at the beginning of each file).

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

  #!/usr/bin/env python
  import limbo
  def build(bld):

    limbo.create_variants(bld,
                        source = 'multi.cpp',
                        uselib_local = 'limbo',
                        uselib = 'BOOST EIGEN TBB SFERES',
                        variants = ['PAREGO',
                                    'EHVI'])


Limbo will create two binaries:

- ``multi_parego``, which is the compilation of ``multi.cpp`` file with a ``#define PAREGO`` at the first line
- ``multi_ehvi``, which is the compilation of ``multi.cpp`` file with a ``#define EHVI`` at the first line

You can add as many defines as you like (or even generate them with python code), for instance:


.. code-block:: python

  #!/usr/bin/env python
  import limbo
  def build(bld):

    limbo.create_variants(bld,
                        source = 'multi.cpp',
                        uselib_local = 'limbo',
                        uselib = 'BOOST EIGEN TBB SFERES',
                        variants = ['PAREGO MOP2 DIM2',
                                    'EHVI ZDT2 DIM6'])


This will create ``multi_parego_mop2_dim2`` and ``multi_ehvi_zdt2_dim6``.

Using ``./waf --exp your_experiment`` will compile all the corresponding libraries. If you want to compile a single variant, you can use the ``--target`` option: ``./waf --exp your_experiment --target parego_mop2_dim2``.

If you have more than one file, you have 2 options:

- First compile a static library, then link with it in the variant.
- Add them in sequence in the source input. The name of the first file is used for the variant target names. Example:

.. code-block:: python

  #!/usr/bin/env python
  import limbo
  def build(bld):

    limbo.create_variants(bld,
                        source = 'multi.cpp dep.cpp impl.cpp',
                        uselib_local = 'limbo',
                        uselib = 'BOOST EIGEN TBB SFERES',
                        variants = ['PAREGO',
                                    'EHVI'])

Limbo will create two binaries:

- ``multi_parego``, which is the compilation of ``multi.cpp``, ``dep.cpp`` and ``impl.cpp`` files with a ``#define PAREGO`` at the first line of each file
- ``multi_ehvi``, which is the compilation of ``multi.cpp``, ``dep.cpp`` and ``impl.cpp`` files with a ``#define EHVI`` at the first line of each file
