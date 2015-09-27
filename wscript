#!/usr/bin/env python
# encoding: utf-8

VERSION = '0.0.1'
APPNAME = 'limbo'

srcdir = '.'
blddir = 'build'

import glob
import limbo


def options(opt):
        opt.load('compiler_cxx boost waf_unit_test')
        opt.load('compiler_c')
        opt.load('eigen')
        opt.load('tbb')
        opt.load('mkl')
        opt.load('sferes')
        opt.load('limbo')
        opt.load('openmp')
        opt.add_option('--exp', type='string', help='exp(s) to build, separate by comma', dest='exp')
        opt.add_option('--qsub', type='string', help='config file (json) to submit to torque', dest='qsub')
        opt.add_option('--oar', type='string', help='config file (json) to submit to oar', dest='oar')
        opt.load('xcode')
        for i in glob.glob('exp/*'):
                opt.recurse(i)


def configure(conf):
        conf.load('compiler_cxx boost waf_unit_test')
        conf.load('compiler_c')
        conf.load('eigen')
        conf.load('tbb')
        conf.load('mkl')
        conf.load('sferes')
        conf.load('openmp')
        conf.load('xcode')

        if conf.env.CXX_NAME in ["icc", "icpc"]:
            common_flags = "-Wall -std=c++11"
            opt_flags = " -O3 -xHost  -march=native -mtune=native -unroll -fma -g"
        else:
            if int(conf.env['CC_VERSION'][0]+conf.env['CC_VERSION'][1]) < 47:
                common_flags = "-Wall -std=c++0x"
            else:
                common_flags = "-Wall -std=c++11"
            opt_flags = " -O3 -march=native -g"

        conf.check_boost(lib='serialization filesystem \
            system unit_test_framework program_options \
            graph thread', min_version='1.39')
        conf.check_eigen()
        conf.check_tbb()
        conf.check_openmp()
        conf.check_mkl()
        conf.check_sferes()
        if conf.is_defined('USE_TBB'):
                common_flags += " -DUSE_TBB"

        if conf.is_defined('USE_SFERES'):
                common_flags += " -DUSE_SFERES -DSFERES_FAST_DOMSORT"

        all_flags = common_flags + opt_flags
        conf.env['CXXFLAGS'] = conf.env['CXXFLAGS'] + all_flags.split(' ')
        print conf.env['CXXFLAGS']

        if conf.options.exp:
                for i in conf.options.exp.split(','):
                        print 'configuring for exp: ' + i
                        conf.recurse('exp/' + i)


def build(bld):
    bld.recurse('src/')
    if bld.options.exp:
        for i in bld.options.exp.split(','):
            print 'Building exp: ' + i
            bld.recurse('exp/' + i)
        from waflib.Tools import waf_unit_test
        bld.add_post_fun(waf_unit_test.summary)


def shutdown(ctx):
    if ctx.options.qsub:
        limbo.qsub(ctx.options.qsub)
    if ctx.options.oar:
        limbo.oar(ctx.options.oar)
