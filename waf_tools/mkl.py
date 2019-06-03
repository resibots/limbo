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
#|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
#|   - Federico Allocati (fede.allocati@gmail.com)
#|   - Vaios Papaspyros (b.papaspyros@gmail.com)
#|   - Roberto Rama (bertoski@gmail.com)
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
#! /usr/bin/env python
# F Allocati - 2015

"""
Quick n dirty intel mkl detection
"""

import os, glob, types
from waflib.Configure import conf
import limbo

def options(opt):
    opt.add_option('--mkl', type='string', help='path to Intel Math Kernel Library', dest='mkl')


@conf
def check_mkl(conf):
    if conf.options.mkl:
        includes_mkl = [conf.options.mkl + '/include']
        libpath_mkl = [conf.options.mkl + '/lib/intel64', conf.options.mkl + '/lib/']
    else:
        includes_mkl = ['/usr/local/include', '/usr/include', '/opt/intel/mkl/include']
        libpath_mkl = ['/usr/local/lib/', '/usr/lib', '/opt/intel/mkl/lib/intel64', '/usr/lib/x86_64-linux-gnu/', '/opt/intel/mkl/lib']

    conf.start_msg('Checking Intel MKL includes (optional)')
    try:
        res = conf.find_file('mkl.h', includes_mkl)
        conf.end_msg('ok')
        conf.start_msg('Checking Intel MKL libs (optional)')
        limbo.check_lib(conf, 'libmkl_core', libpath_mkl)
        conf.end_msg('ok')
    except:
        conf.end_msg('Not found', 'RED')
        return
    conf.env.LIB_MKL_SEQ = ["mkl_intel_lp64", "mkl_core", "mkl_sequential", "pthread", "m"]
    conf.env.LIB_MKL_TBB = ["mkl_intel_lp64", "mkl_core", "mkl_tbb_thread", "tbb", "stdc++", "pthread", "m"]
    if conf.env.CXX_NAME in ["icc", "icpc"]:
        conf.env.LIB_MKL_OMP = ["mkl_intel_lp64", "mkl_core", "mkl_intel_thread", "pthread", "m"]
    else:
        conf.env.LIB_MKL_OMP = ["mkl_intel_lp64", "mkl_core", "mkl_gnu_thread", "dl", "pthread", "m"]
    conf.env.INCLUDES_MKL_SEQ = includes_mkl
    conf.env.INCLUDES_MKL_TBB = includes_mkl
    conf.env.INCLUDES_MKL_OMP = includes_mkl
    conf.env.LIBPATH_MKL_SEQ = libpath_mkl
    conf.env.LIBPATH_MKL_TBB = libpath_mkl
    conf.env.LIBPATH_MKL_OMP = libpath_mkl
    conf.env.CXXFLAGS_MKL_SEQ = ["-m64",  "-DEIGEN_USE_MKL_ALL", "-DMKL_BLAS=MKL_DOMAIN_BLAS"]
    #conf.env.LINKFLAGS_MKL_SEQ = [ "-Wl,--no-as-needed" ]
    conf.env.CXXFLAGS_MKL_TBB = ["-m64",  "-DEIGEN_USE_MKL_ALL" , "-DMKL_BLAS=MKL_DOMAIN_BLAS"]
    #conf.env.LINKFLAGS_MKL_TBB = [ "-Wl,--no-as-needed" ]
    if  conf.env.CXX_NAME in ["icc", "icpc"]:
        conf.env.CXXFLAGS_MKL_OMP = ["-qopenmp", "-m64",  "-DEIGEN_USE_MKL_ALL", "-DMKL_BLAS=MKL_DOMAIN_BLAS" ]
    else:
        conf.env.CXXFLAGS_MKL_OMP = ["-fopenmp", "-m64",  "-DEIGEN_USE_MKL_ALL", "-DMKL_BLAS=MKL_DOMAIN_BLAS"]
    #conf.env.LINKFLAGS_MKL_OMP = [ "-Wl,--no-as-needed" ]
