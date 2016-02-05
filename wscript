#!/usr/bin/env python
# encoding: utf-8
import sys
sys.path.insert(0, './waf_tools')

VERSION = '0.0.1'
APPNAME = 'limbo'

srcdir = '.'
blddir = 'build'

import glob
import os
import subprocess
import limbo
from waflib.Build import BuildContext

import inspect


def options(opt):
        opt.load('compiler_cxx boost waf_unit_test')
        opt.load('compiler_c')
        opt.load('eigen')
        opt.load('tbb')
        opt.load('mkl')
        opt.load('sferes')
        opt.load('limbo')
        opt.load('openmp')
        opt.load('nlopt')
        opt.load('libcmaes')

        opt.add_option('--exp', type='string', help='exp(s) to build, separate by comma', dest='exp')
        opt.add_option('--qsub', type='string', help='config file (json) to submit to torque', dest='qsub')
        opt.add_option('--oar', type='string', help='config file (json) to submit to oar', dest='oar')
        opt.add_option('--experimental', action='store_true', help='specify to compile the experimental examples', dest='experimental')
        opt.load('xcode')
        for i in glob.glob('exp/*'):
                if os.path.isdir(i):
                    opt.recurse(i)


def configure(conf):
        conf.load('compiler_cxx boost waf_unit_test')
        conf.load('compiler_c')
        conf.load('eigen')
        conf.load('tbb')
        conf.load('sferes')
        conf.load('openmp')
        conf.load('mkl')
        conf.load('xcode')
        conf.load('nlopt')
        conf.load('libcmaes')

        if conf.env.CXX_NAME in ["icc", "icpc"]:
            common_flags = "-Wall -std=c++11"
            opt_flags = " -O3 -xHost  -march=native -mtune=native -unroll -fma -g"
        else:
            if int(conf.env['CC_VERSION'][0]+conf.env['CC_VERSION'][1]) < 47:
                common_flags = "-Wall -std=c++0x"
            else:
                common_flags = "-Wall -std=c++11"
            common_flags += " -fdiagnostics-color"
            opt_flags = " -O3 -march=native -g"

        conf.check_boost(lib='serialization filesystem \
            system unit_test_framework program_options \
            graph thread', min_version='1.39')
        conf.check_eigen()
        conf.check_tbb()
        conf.check_sferes()
        conf.check_openmp()
        conf.check_mkl()
        conf.check_nlopt()
        conf.check_libcmaes()

        conf.env.INCLUDES_LIMBO = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/src"

        if conf.env['CXXFLAGS_ODE']:
                common_flags += ' ' + conf.env['CXXFLAGS_ODE']

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


def build_extensive_tests(ctx):
    ctx.recurse('src/')
    ctx.recurse('src/tests')

def run_extensive_tests(ctx):
    for fullname in glob.glob('build/src/tests/combinations/*'):
        if os.path.isfile(fullname) and os.access(fullname, os.X_OK):
            fpath, fname = os.path.split(fullname)
            print "Running: " + fname
            s = "cd " + fpath + "; ./" + fname
            retcode = subprocess.call(s, shell=True, env=None)

def submit_extensive_tests(ctx):
    for fullname in glob.glob('build/src/tests/combinations/*'):
        if os.path.isfile(fullname) and os.access(fullname, os.X_OK):
            fpath, fname = os.path.split(fullname)
            s = "cd " + fpath + ";oarsub -l /nodes=1/core=2,walltime=00:15:00 -n " + fname + " -O " + fname + ".stdout.%jobid%.log -E " + fname + ".stderr.%jobid%.log ./" + fname
            retcode = subprocess.call(s, shell=True, env=None)
            print "oarsub returned:" + str(retcode)


def shutdown(ctx):
    if ctx.options.qsub:
        limbo.qsub(ctx.options.qsub)
    if ctx.options.oar:
        limbo.oar(ctx.options.oar)


class BuildExtensiveTestsContext(BuildContext):
    cmd = 'build_extensive_tests'
    fun = 'build_extensive_tests'
