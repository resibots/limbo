#! /usr/bin/env python
# encoding: utf-8
# to be put in your experiment's directory

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
