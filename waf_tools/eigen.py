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
# JB Mouret - 2009

"""
Quick n dirty eigen3 detection
"""

import os, glob, types
import subprocess
from waflib.Configure import conf


def options(opt):
	opt.add_option('--eigen', type='string', help='path to eigen', dest='eigen')


@conf
def check_eigen(conf):
	conf.start_msg('Checking for Eigen')
	includes_check = ['/usr/include/eigen3', '/usr/local/include/eigen3', '/usr/include', '/usr/local/include']

	if conf.options.eigen:
		includes_check = [conf.options.eigen]

	try:
		res = conf.find_file('Eigen/Core', includes_check)
		incl = res[:-len('Eigen/Core')-1]
		conf.env.INCLUDES_EIGEN = [incl]
		p1 = subprocess.Popen(["cat", incl+"/Eigen/src/Core/util/Macros.h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		p2 = subprocess.Popen(["grep", "#define EIGEN_WORLD_VERSION"], stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		p1.stdout.close()
		out1, err = p2.communicate()
		p1 = subprocess.Popen(["cat", incl+"/Eigen/src/Core/util/Macros.h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		p2 = subprocess.Popen(["grep", "#define EIGEN_MAJOR_VERSION"], stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		p1.stdout.close()
		out2, err = p2.communicate()
		world_version = int(out1.strip()[-1])
		major_version = int(out2.strip()[-1])
		if world_version == 3 and major_version >= 3:
			# Check for lapacke and blas
			extra_libs = ['/usr/lib', '/usr/local/lib']
			blas_libs = ['blas', 'openblas']
			try:
				blas_lib = ''
				for b in blas_libs:
					conf.find_file('lib'+b+'.so', extra_libs)
					blas_lib = b
					break
				conf.find_file('liblapacke.so', extra_libs)
				conf.env.DEFINES_EIGEN = ['EIGEN_USE_LAPACKE', 'EIGEN_USE_BLAS']
				conf.env.LIBPATH_EIGEN = extra_libs
				conf.env.LIB_EIGEN = ['lapacke', blas_lib]
			except:
				pass
		conf.end_msg('ok')
	except:
		conf.end_msg('Not found', 'RED')
	return 1
