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
		conf.env.INCLUDES_SFERES = [conf.options.sferes]
		conf.env.LIBPATH_SFERES = [conf.options.sferes + '/build/default/sferes']
        try:
                res = conf.find_file('sferes/ea/ea.hpp', conf.env.INCLUDES_SFERES)
                conf.define("USE_SFERES", 1)
        except:
                print 'SFERES not found'
	return 1


