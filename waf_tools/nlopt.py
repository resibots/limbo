#! /usr/bin/env python
# encoding: utf-8
# Konstantinos Chatzilygeroudis - 2015

"""
Quick n dirty nlopt detection
"""

import os, glob, types, sys
from waflib.Configure import conf


def options(opt):
	opt.add_option('--nlopt', type='string', help='path to nlopt', dest='nlopt')


@conf
def check_nlopt(conf):
	conf.start_msg('Checking for NLOpt')
	if conf.options.nlopt:
		conf.env.INCLUDES_NLOPT = [conf.options.nlopt + '/include']
		conf.env.LIBPATH_NLOPT = [conf.options.nlopt + '/lib']
		conf.env.LIB_NLOPT = ['nlopt_cxx']
	else:
		conf.env.INCLUDES_NLOPT = [os.environ['RESIBOTS_DIR'] + '/include', '/usr/local/include']
        conf.env.LIBPATH_NLOPT = [os.environ['RESIBOTS_DIR'] + '/lib', '/usr/local/lib']
        conf.env.LIB_NLOPT = ['nlopt_cxx']

	try:
		res = conf.find_file('nlopt.hpp', conf.env.INCLUDES_NLOPT)
		conf.define("USE_NLOPT", 1)
		conf.end_msg('ok')
	except:
		conf.end_msg('Not found', 'RED')
		return
	return 1
