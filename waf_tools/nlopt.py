#! /usr/bin/env python
# encoding: utf-8
# Konstantinos Chatzilygeroudis - 2015

"""
Quick n dirty nlopt detection
"""

import os, glob, types
from waflib.Configure import conf


def options(opt):
	opt.add_option('--nlopt', type='string', help='path to nlopt', dest='nlopt')


@conf
def check_nlopt(conf):
	includes_check = ['/usr/local/include/robdyn']
	libs_check = ['/usr/local/lib']
	if 'RESIBOTS_DIR' in os.environ:
		includes_check = [os.environ['RESIBOTS_DIR'] + '/include'] + includes_check
		libs_check = [os.environ['RESIBOTS_DIR'] + '/lib'] + libs_check

	if conf.options.nlopt:
		conf.env.INCLUDES_NLOPT = [conf.options.nlopt + '/include']
		conf.env.LIBPATH_NLOPT = [conf.options.nlopt + '/lib']
	else:
		conf.env.INCLUDES_NLOPT = includes_check
		conf.env.LIBPATH_NLOPT = libs_check

	try:
		conf.start_msg('Checking for NLOpt includes')
		res = conf.find_file('nlopt.hpp', conf.env.INCLUDES_NLOPT)
		conf.end_msg('ok')
		conf.start_msg('Checking for NLOpt libs')
		res = res and conf.find_file('libnlopt_cxx.so', conf.env.LIBPATH_NLOPT)
		conf.end_msg('ok')
		conf.env.DEFINES_NLOPT = ['USE_NLOPT']
		conf.env.LIB_NLOPT = ['nlopt_cxx']
	except:
		conf.end_msg('Not found', 'RED')
		return
	return 1
