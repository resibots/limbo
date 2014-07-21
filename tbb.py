#! /usr/bin/env python
# encoding: utf-8
# JB Mouret - 2014

"""
Quick n dirty tbb detection
"""

import os, glob, types
from waflib.Configure import conf

def options(opt):
	opt.add_option('--tbb', type='string', help='path to Intel TBB', dest='tbb')


@conf
def check_tbb(conf):
        conf.env.LIB_TBB = ['tbb']
	if conf.options.tbb:
		conf.env.INCLUDES_TBB = [conf.options.tbb + '/include']
		conf.env.LIBPATH_TBB = [conf.options.tbb + '/lib']
	else:
		conf.env.INCLUDES_TBB = ['/usr/local/include',
                                           '/usr/include']
		conf.env.LIBPATH_TBB = ['/usr/local/lib/',
                                           '/usr/lib']

        try:
                res = conf.find_file('tbb/parallel_for.h', conf.env.INCLUDES_TBB)
                conf.define("USE_TBB", 1)
        except:
                print 'TBB not found'


