Add External Library
====================

Add external library to the wscript of your experiment
--------------------------------------------

To add an external library to your experiment, you need to modify the build script of your experiment, named ``wscript``. The standard way to do this is to create a new configuration file for the new dependency. In the ``waf`` build system, this is done by creating a python script (``.py`` file), usually called ``libname.py`` (where ``libname`` is the name of the library), in the same directory as your experiment.

.. warning:: to activate this script, you need to activate your experiment **when configuring limbo**::

  ./waf configure --exp your_exp

This new file should have the following structure:

.. code:: python

    #!/usr/bin/env python
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

Where ``check_if_libname_exists`` is replaced with logic to find our library, as explained later. If the library is optional, the ``conf.fatal`` line can be removed.

Then add the following lines in the ``wscript``:

.. code:: python

    # imports, etc, ...

    # we assume that the configuration file is saved as libname.py
    import libname

    def configure(conf):
        conf.load('libname')
        conf.check_libname()
        # rest of the configuration

    # rest of code


Libraries usually have the headers and the lib files in two different directories. However, header-only libraries only have includes, in this case, you can ignore the following section named "Check for lib files".

Check for headers
^^^^^^^^^^^^^^^^^

To check for the headers of the library, you can add the following code to the ``check_libname`` function:

.. code:: python

    # previous code

    @conf
    def check_libname(conf):
        # possible path to find headers
        includes_check = ['path1', 'path2']
        try:
          conf.start_msg('Checking for libname includes')
          # include_files is a list with the headers we expect to find
          for file in include_files:
            conf.find_file(file, includes_check)
          conf.end_msg('ok')
          conf.env.INCLUDES_LIBNAME = includes_check
        except:
          conf.end_msg('Not found', 'RED')
          return
        # rest of check_libname

    # rest of code

Check for lib files
^^^^^^^^^^^^^^^^^^^^

To check for the lib files of the library, you can add the following code to the ``check_libname`` function:

.. code:: python

    # previous code

    @conf
    def check_libname(conf):
        # possible path to find lib files
        libs_check = ['path1', 'path2']
        try:
          conf.start_msg('Checking for libname libs')
          # lib_files is a list with the lib files we expect to find
          for file in lib_files:
            conf.find_file(file, libs_check)
          conf.end_msg('ok')
          conf.env.LIBPATH_LIBNAME = libs_check
          # list with the lib names the library has
          conf.env.LIB_LIBNAME = ['libname1', 'libname2']
        except:
          conf.end_msg('Not found', 'RED')
          return
        # rest of check_libname

    # rest of code

Add configuration options
^^^^^^^^^^^^^^^^^^^^^^^^^

Additional configuration options are often needed when adding new libraries. For example, one useful option is to specify the location of the library headers and lib files. Adding options is easy: you only need to define a new function named ``options`` in the ``wscript`` and another one in the library configuration file. In the library's configuration file (e.g., ``libname.py``):

.. code:: python

    #imports, etc, ...

    def options(opt):
        # add options to the configuration
        opt.add_option('cmd_option', type='option_type', help='info message',
            dest='destination_variable')

    @conf
    def check_libname(conf):
        # access options
        if conf.options.destination_variable == 'yes':
          print 'destination_variable found'
        # rest of check_libname

The options in the waf build system are using the python's ``optparse``. Check the official `optparse`_ documentation for more information.

.. _optparse: https://docs.python.org/2/library/optparse.html

Then add the following lines in the ``wscript`` of your experiment:

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
  |-- src
  ...

**wscript:**

.. code:: python

    #!/usr/bin/env python

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

    #!/usr/bin/env python
    # encoding: utf-8

    import os
    from waflib.Configure import conf


    def options(opt):
      opt.add_option('--ros', type='string', help='path to ros', dest='ros')

    @conf
    def check_ros(conf):
      # Get locations where to search for ROS's header and lib files
      if conf.options.ros:
        includes_check = [conf.options.ros + '/include']
        libs_check = [conf.options.ros + '/lib']
      else:
        if 'ROS_DISTRO' not in os.environ:
          conf.start_msg('Checking for ROS')
          conf.end_msg('ROS_DISTRO not in environmental variables', 'RED')
          return
        includes_check = ['/opt/ros/' + os.environ['ROS_DISTRO'] + '/include']
        libs_check = ['/opt/ros/' + os.environ['ROS_DISTRO'] + '/lib/']

      try:
        # Find the header for ROS
        conf.start_msg('Checking for ROS includes')
        conf.find_file('ros/ros.h', includes_check)
        conf.end_msg('ok')

        # Find the lib files
        libs = ['roscpp','rosconsole','roscpp_serialization','rostime','xmlrpcpp',
                'rosconsole_log4cxx', 'rosconsole_backend_interface']
        conf.start_msg('Checking for ROS libs')
        for lib in libs:
          conf.find_file('lib'+lib+'.so', libs_check)
        conf.end_msg('ok')

        conf.env.INCLUDES_ROS = includes_check
        conf.env.LIBPATH_ROS = libs_check
        conf.env.LIB_ROS = libs
        conf.env.DEFINES_ROS = ['USE_ROS']
      except:
        conf.end_msg('Not found', 'RED')
        return

The configuration and compilation of the experiment follows the usual procedure (assuming that we are in the **limbo** root folder)::

  ./waf configure --exp example
  ./waf --exp example
