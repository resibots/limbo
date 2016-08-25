#!/usr/bin/env python
# encoding: utf-8
#| Copyright Inria May 2015
#| This project has received funding from the European Research Council (ERC) under
#| the European Union's Horizon 2020 research and innovation programme (grant
#| agreement No 637972) - see http://www.resibots.eu
#| 
#| Contributor(s):
#|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
#|   - Antoine Cully (antoinecully@gmail.com)
#|   - Kontantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
#|   - Federico Allocati (fede.allocati@gmail.com)
#|   - Vaios Papaspyros (b.papaspyros@gmail.com)
#| 
#| This software is a computer library whose purpose is to optimize continuous,
#| black-box functions. It mainly implements Gaussian processes and Bayesian
#| optimization.
#| Main repository: http://github.com/resibots/limbo
#| Documentation: http://www.resibots.eu/limbo
#| 
#| This software is governed by the CeCILL-C license under French law and
#| abiding by the rules of distribution of free software.  You can  use,
#| modify and/ or redistribute the software under the terms of the CeCILL-C
#| license as circulated by CEA, CNRS and INRIA at the following URL
#| "http://www.cecill.info".
#| 
#| As a counterpart to the access to the source code and  rights to copy,
#| modify and redistribute granted by the license, users are provided only
#| with a limited warranty  and the software's author,  the holder of the
#| economic rights,  and the successive licensors  have only  limited
#| liability.
#| 
#| In this respect, the user's attention is drawn to the risks associated
#| with loading,  using,  modifying and/or developing or reproducing the
#| software by the user in light of its specific status of free software,
#| that may mean  that it is complicated to manipulate,  and  that  also
#| therefore means  that it is reserved for developers  and  experienced
#| professionals having in-depth computer knowledge. Users are therefore
#| encouraged to load and test the software's suitability as regards their
#| requirements in conditions enabling the security of their systems and/or
#| data to be ensured and,  more generally, to use and operate it in the
#| same conditions as regards security.
#| 
#| The fact that you are presently reading this means that you have had
#| knowledge of the CeCILL-C license and that you accept its terms.
#| 
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
import inspect
from waflib.Build import BuildContext

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
        opt.add_option('--local', type='string', help='config file (json) to run local', dest='local')
        opt.add_option('--local_serial', type='string', help='config file (json) to run local', dest='local_serial')
        opt.add_option('--experimental', action='store_true', help='specify to compile the experimental examples', dest='experimental')
        opt.add_option('--nb_replicates', type='int', help='number of replicates performed during the benchmark', dest='nb_rep')
        # tests
        opt.add_option('--tests', action='store_true', help='compile tests or not', dest='tests')
        opt.load('xcode')
        for i in glob.glob('exp/*'):
                if os.path.isdir(i):
                    opt.recurse(i)

        opt.recurse('src/benchmarks')

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
            if conf.env.CXX_NAME in ["gcc", "g++"] and int(conf.env['CC_VERSION'][0]+conf.env['CC_VERSION'][1]) < 47:
                common_flags = "-Wall -std=c++0x"
            else:
                common_flags = "-Wall -std=c++11"
            if conf.env.CXX_NAME in ["clang", "llvm"]:
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

        all_flags = common_flags + opt_flags
        conf.env['CXXFLAGS'] = conf.env['CXXFLAGS'] + all_flags.split(' ')
        print conf.env['CXXFLAGS']

        if conf.options.exp:
                for i in conf.options.exp.split(','):
                        print 'configuring for exp: ' + i
                        conf.recurse('exp/' + i)
        conf.recurse('src/benchmarks')

def build(bld):
    bld.recurse('src/')
    if bld.options.exp:
        for i in bld.options.exp.split(','):
            print 'Building exp: ' + i
            bld.recurse('exp/' + i)
            limbo.output_params('exp/'+i)
    bld.add_post_fun(limbo.summary)

def build_extensive_tests(ctx):
    ctx.recurse('src/')
    ctx.recurse('src/tests')

def build_benchmark(ctx):
    ctx.recurse('src/benchmarks')

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

def run_benchmark(ctx):
    HEADER='\033[95m'
    NC='\033[0m'
    res_dir=os.getcwd()+"/benchmark_results/"
    try:
        os.makedirs(res_dir)
    except:
        print "WARNING, dir:" + res_dir + " not be created"
    for fullname in glob.glob('build/src/benchmarks/*'):
        if os.path.isfile(fullname) and os.access(fullname, os.X_OK):
            fpath, fname = os.path.split(fullname)
            directory = res_dir + "/" + fname
            try:
                os.makedirs(directory)
            except:
                print "WARNING, dir:" + directory + " not be created, the new results will be concatenated to the old ones"
            s = "cp " + fullname + " " + directory
            retcode = subprocess.call(s, shell=True, env=None)
            if ctx.options.nb_rep:
                nb_rep = ctx.options.nb_rep
            else:
                nb_rep = 10
            for i in range(0,nb_rep):
                print HEADER+" Running: " + fname + " for the "+str(i)+"th time"+NC
                s="cd " + directory +";./" + fname
                retcode = subprocess.call(s, shell=True, env=None)

def shutdown(ctx):
    if ctx.options.qsub:
        limbo.qsub(ctx.options.qsub)
    if ctx.options.oar:
        limbo.oar(ctx.options.oar)
    if ctx.options.local:
        limbo.run_local(ctx.options.local, False)
    if ctx.options.local_serial:
        limbo.run_local(ctx.options.local_serial)

def insert_license(ctx):
    limbo.insert_license()

class BuildExtensiveTestsContext(BuildContext):
    cmd = 'build_extensive_tests'
    fun = 'build_extensive_tests'

class BuildBenchmark(BuildContext):
    cmd = 'build_benchmark'
    fun = 'build_benchmark'

class InsertLicense(BuildContext):
    cmd = 'insert_license'
    fun = 'insert_license'
