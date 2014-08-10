#! /usr/bin/env python
# encoding: utf-8
# JB Mouret - 2009

"""
Quick n dirty sferes2 detection
"""

import os, glob, types
from waflib.Configure import conf


def options(opt):
	opt.add_option('--sferes', type='string', help='path to sferes2', dest='sferes')


@conf
def check_sferes(conf):
	if conf.options.sferes:
		conf.env.INCLUDES_EIGEN = [conf.options.sferes]
		conf.env.LIBPATH_EIGEN = [conf.options.sferes + '/build/default/sferes']
	return 1


