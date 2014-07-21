#!/usr/bin/env python
# encoding: utf-8

VERSION='0.0.1'
APPNAME='limbo'

srcdir = '.'
blddir = 'build'

import copy
import os, sys

def options(opt):
        opt.load('compiler_cxx boost waf_unit_test')
        opt.load('compiler_c')
        opt.load('eigen')
        opt.load('tbb')

def configure(conf):
    	print("configuring b-optimize")
    	conf.load('compiler_cxx boost waf_unit_test')
        conf.load('compiler_c')
        conf.load('eigen')
        conf.load('tbb')

	common_flags = "-Wall -std=c++11 -fcolor-diagnostics"

	cxxflags = conf.env['CXXFLAGS']
	conf.check_boost(lib='serialization timer filesystem system unit_test_framework program_options graph mpi python thread',
			 min_version='1.35')
        conf.check_eigen()
        conf.check_tbb()
        if conf.is_defined('USE_TBB'):
                common_flags += " -DUSE_TBB "

	# release
        opt_flags = common_flags + ' -O3 -msse2 -ggdb3'
        conf.env['CXXFLAGS'] = cxxflags + opt_flags.split(' ')
        print conf.env['CXXFLAGS']

def build(bld):
	bld.recurse('src/limbo')
        bld.recurse('src/examples')
        bld.recurse('src/tests')
        from waflib.Tools import waf_unit_test
        bld.add_post_fun(waf_unit_test.summary)
