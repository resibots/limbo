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
# JB Mouret - 2014

"""
Quick n dirty tbb detection
"""

from waflib.Configure import conf


def options(opt):
    opt.add_option('--tbb', type='string', help='path to Intel TBB', dest='tbb')


@conf
def check_tbb(self, *k, **kw):
    if self.options.tbb:
        includes_tbb = [self.options.tbb + '/include']
        libpath_tbb = [self.options.tbb + '/lib']
    else:
        includes_tbb = ['/usr/local/include', '/usr/include', '/opt/intel/tbb/include']
        libpath_tbb = ['/usr/local/lib/', '/usr/lib', '/opt/intel/tbb/lib']

    self.start_msg('Checking Intel TBB includes (optional)')
    try:
        self.find_file('tbb/parallel_for.h', includes_tbb)
        self.end_msg('ok')
    except:
        self.end_msg('not found', 'YELLOW')
        return

    self.start_msg('Checking Intel TBB libs (optional)')
    try:
        self.find_file('libtbb.so', libpath_tbb)
        self.end_msg('ok')
    except:
        self.end_msg('not found', 'YELLOW')
        return

    self.env.LIBPATH_TBB = libpath_tbb
    self.env.LIB_TBB = ['tbb']
    self.env.INCLUDES_TBB = includes_tbb
    self.env.DEFINES_TBB = ['USE_TBB']
