Add External Library
====================

Add external library to experiment's wscript
--------------------------------------------

To add an external library to your experiment, we need to modify our experiment's ``wscript``. The stantard way to do this is to create a new configuration file for the dependency we want. In the ``waf`` build system, we do this by creating a python script(`.py`), usually called ``libname.py`` (where ``libname`` is the name of the library), in the same directory as your experiment.

.. warning:: to activate this script, you need to activate your experiment **when configuring limbo**:  ``./waf configure --exp your_exp``

This new file should have the following structure:

.. code:: python

    #! /usr/bin/env python
    # encoding: utf-8

    from waflib.Configure import conf

    @conf
    def check_libname(conf):
        # check if libname exists in the system
        try:
          res = check_if_libname_exists
        except:
          conf.fatal('libname not found')
          return
        return 1

Where we replace the ``check_if_libname_exists`` with logic to find our library. If we want the library to be optional, we omit the ``conf.fatal`` part.

Then in our ``wscript`` we add the following lines:

.. code:: python

    # imports, etc, ...

    # we assume that the configuration file is saved as libname.py
    import libname

    def configure(conf):
        conf.load('libname')
        conf.check_libname()
        # rest of the configuration

    # rest of code


Libraries usually have the headers and the lib files in two different directories. Header-only libraries only have includes.

Check for headers
^^^^^^^^^^^^^^^^^

To check for the headers of the library, we add the following code to the ``check_libname`` function:

.. code:: python

    # previous code

    @conf
    def check_libname(conf):
        # possible path to find headers
        includes_check = ['path1', 'path2']
        try:
          conf.start_msg('Checking for libname includes')
          res = True
          # include_files is a list with the headers we expect to find
          for file in include_files:
            res = res and conf.find_file(file, includes_check)
          conf.end_msg('ok')
          conf.env.INCLUDES_LIBNAME = includes_check
        except:
          conf.end_msg('Not found', 'RED')
          return
        # rest of check_libname

    # rest of code

Check for lib files
^^^^^^^^^^^^^^^^^^^^

To check for the lib files of the library, we add the following code to the ``check_libname`` function:

.. code:: python

    # previous code

    @conf
    def check_libname(conf):
        # possible path to find lib files
        libs_check = ['path1', 'path2']
        try:
          conf.start_msg('Checking for libname libs')
          res = True
          # lib_files is a list with the lib files we expect to find
          for file in lib_files:
            res = res and conf.find_file(file, libs_check)
          conf.end_msg('ok')
          conf.env.LIBPATH_LIBNAME = libs_check
          # list with the lib names the library has
          conf.env.LIB_LIBNAME = ['libname1', 'libname2']
        except:
          conf.end_msg('Not found', 'RED')
          return
        # rest of check_libname

    # rest of code

Add options
^^^^^^^^^^^^

We often need specific options when adding new libraries. One useful option, for example, is to specify where to find the library headers and lib files. Adding options is easy: we only need to add a new function named ``options`` in our ``wscript`` and another one in the library configuration file:

.. code:: python

    #imports, etc, ...

    def options(opt):
        # add options to the configuration
        opt.add_option('cmd_option', type='option_type', help='info message', dest='destination_variable')

    @conf
    def check_libname(conf):
        # access options
        if conf.options.destination_variable == 'yes':
          print 'destination_variable found'
        # rest of check_libname

The options in the waf build system are using the python's ``optparse``. Check the official `optparse`_ documentation for more information.

.. _optparse: https://docs.python.org/2/library/optparse.html

Then in our ``wscript`` we add the following lines:

.. code:: python

    # imports, etc, ...

    def options(opt):
        opt.load('libname')
        # rest of the options

    # rest of the code


Example: Add ROS as external library
-------------------------------------

Here's a small and quick example to add `ROS`_ as an external library to our experiment. We assume the following file structure (where ``main.cpp`` is C++ source code using **limbo** and **ROS**):

.. _ROS: http://www.ros.org/

::

  limbo
  |-- exp
       |-- example
            +-- wscript
            +-- ros.py
            +-- main.cpp

**wscript:**

.. code:: python

    #! /usr/bin/env python

    import limbo
    import ros

    def options(opt):
        opt.load('ros')

    def configure(conf):
        conf.load('ros')
        conf.check_ros()

    def build(bld):
        libs = 'EIGEN BOOST ROS LIMBO'

        obj = bld(features = 'cxx cxxstlib',
                  source = 'main.cpp',
                  includes = '. .. ../../ ../../src',
                  target = 'test_exec',
                  uselib =  libs,
                  use = 'limbo')

**ros.py:**

.. code:: python

    #! /usr/bin/env python
    # encoding: utf-8

    import os
    from waflib.Configure import conf


    def options(opt):
      opt.add_option('--ros', type='string', help='path to ros', dest='ros')

    @conf
    def check_ros(conf):
      if conf.options.ros:
        includes_check = [conf.options.ros + '/include']
        libs_check = [conf.options.ros + '/lib']
      else:
        if 'ROS_DISTRO' not in os.environ:
          conf.start_msg('Checking for ROS')
          conf.end_msg('ROS_DISTRO not in environmental variables', 'RED')
          return 1
        includes_check = ['/opt/ros/' + os.environ['ROS_DISTRO'] + '/include']
        libs_check = ['/opt/ros/' + os.environ['ROS_DISTRO'] + '/lib/']

      try:
        conf.start_msg('Checking for ROS includes')
        res = conf.find_file('ros/ros.h', includes_check)
        conf.end_msg('ok')
        libs = ['roscpp','rosconsole','roscpp_serialization','rostime', 'xmlrpcpp','rosconsole_log4cxx', 'rosconsole_backend_interface']
        conf.start_msg('Checking for ROS libs')
        for lib in libs:
          res = res and conf.find_file('lib'+lib+'.so', libs_check)
        conf.end_msg('ok')
        conf.env.INCLUDES_ROS = includes_check
        conf.env.LIBPATH_ROS = libs_check
        conf.env.LIB_ROS = libs
        conf.env.DEFINES_ROS = ['USE_ROS']
      except:
        conf.end_msg('Not found', 'RED')
        return 1
      return 1

Assuming we are at **limbo** root, we run the following to compile our experiment: ::

  ./waf configure --exp example
  ./waf --exp example
