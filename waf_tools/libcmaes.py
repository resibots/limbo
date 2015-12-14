#! /usr/bin/env python
# encoding: utf-8
# JB Mouret / Inria - 2015

"""
Quick n dirty libcmaes detection
"""

import os, glob, types
from waflib.Configure import conf


def options(opt):
	opt.add_option('--libcmaes', type='string', help='path to libcmaes', dest='libcmaes')


@conf
def check_libcmaes(conf):
	if conf.options.libcmaes:
		includes_check = [conf.options.libcmaes + '/include']
		libs_check = [conf.options.libcmaes + '/lib']
	else:
		includes_check = ['/usr/local/include', '/usr/include']
		libs_check = ['/usr/local/lib', '/usr/lib']

	try:
		conf.start_msg('Checking for libcmaes includes')
		res = conf.find_file('libcmaes/cmaes.h', includes_check)
		conf.end_msg('ok')
	except:
		conf.end_msg('Not found', 'RED')
		return 1
	conf.start_msg('Checking for libcmaes libs')
	found = False
	for lib in ['libcmaes.so', 'libcmaes.a', 'libcmaes.dylib']:
		try:
			found = found or conf.find_file(lib, libs_check)
		except:
			continue
	if not found:
		conf.end_msg('Not found', 'RED')
		return 1
	else:
		conf.end_msg('ok')
		conf.env.INCLUDES_LIBCMAES = includes_check
		conf.env.LIBPATH_LIBCMAES = libs_check
		conf.env.DEFINES_LIBCMAES = ['USE_LIBCMAES']
		conf.env.LIB_LIBCMAES= ['cmaes']
	return 1
