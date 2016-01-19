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
	if conf.options.nlopt:
		includes_check = [conf.options.nlopt + '/include']
		libs_check = [conf.options.nlopt + '/lib']
	else:
		includes_check = ['/usr/local/include', '/usr/include']
		libs_check = ['/usr/local/lib', '/usr/lib', '/usr/lib/x86_64-linux-gnu/']
		if 'RESIBOTS_DIR' in os.environ:
			includes_check = [os.environ['RESIBOTS_DIR'] + '/include'] + includes_check
			libs_check = [os.environ['RESIBOTS_DIR'] + '/lib'] + libs_check

	try:
		conf.start_msg('Checking for NLOpt includes')
		res = conf.find_file('nlopt.hpp', includes_check)
		conf.end_msg('ok')
	except:
		conf.end_msg('Not found', 'RED')
		return 1
	conf.start_msg('Checking for NLOpt libs')
	found = False
	for lib in ['libnlopt_cxx.so', 'libnlopt_cxx.a', 'libnlopt_cxx.dylib']:
		try:
			found = found or conf.find_file(lib, libs_check)
		except:
			continue
	if not found:
		conf.end_msg('Not found', 'RED')
		return 1
	else:
		conf.end_msg('ok')
		conf.env.INCLUDES_NLOPT = includes_check
		conf.env.LIBPATH_NLOPT = libs_check
		conf.env.DEFINES_NLOPT = ['USE_NLOPT']
		conf.env.LIB_NLOPT = ['nlopt_cxx']
	return 1
