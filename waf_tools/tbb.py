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
# JB Mouret - 2014

"""
Quick n dirty tbb detection
"""

from waflib.Configure import conf
import limbo

def options(opt):
    opt.add_option('--tbb', type='string', help='path to Intel TBB', dest='tbb')

@conf
def check_tbb(self, *k, **kw):
    def get_directory(filename, dirs):
        res = self.find_file(filename, dirs)
        return res[:-len(filename)-1]

    required = kw.get('required', False)

    if self.options.tbb:
        includes_tbb = [self.options.tbb + '/include']
        libpath_tbb = [self.options.tbb + '/lib', self.options.tbb + '/lib64']
    else:
        includes_tbb = ['/usr/local/include', '/usr/include', '/opt/local/include', '/sw/include', '/opt/homebrew/include']
        libpath_tbb = ['/usr/lib', '/usr/local/lib64', '/usr/local/lib', '/opt/local/lib', '/sw/lib', '/lib', '/usr/lib64', '/usr/lib/x86_64-linux-gnu/', '/usr/local/lib/x86_64-linux-gnu/', '/usr/lib/aarch64-linux-gnu/', '/usr/local/lib/aarch64-linux-gnu/', '/opt/homebrew/lib']

    opt_msg = ' (optional)' if not required else ''
    self.start_msg('Checking Intel TBB includes' + opt_msg)
    incl = ''
    lib = ''
    try:
        incl = get_directory('tbb/parallel_for.h', includes_tbb)
        self.end_msg(incl)
    except:
        if required:
            self.fatal('Not found in %s' % str(includes_tbb))
        self.end_msg('Not found in %s' % str(includes_tbb), 'YELLOW')
        return

    # check for oneapi vs older tbb
    self.start_msg('Checking for Intel OneAPI TBB' + opt_msg)
    using_oneapi = False
    try:
        incl = get_directory('oneapi/tbb.h', includes_tbb)
        self.end_msg(incl)
        using_oneapi = True
    except:
        self.end_msg('Not found in %s, reverting to older TBB' % str(includes_tbb), 'YELLOW')
        using_oneapi = False


    self.start_msg('Checking Intel TBB libs' + opt_msg)
    try:
        res, lib = limbo.check_lib(self, 'libtbb', libpath_tbb)
        self.end_msg(lib)
    except:
        if required:
            self.fatal('Not found in %s' % str(libpath_tbb))
        self.end_msg('Not found in %s' % str(libpath_tbb), 'YELLOW')

    self.env.LIBPATH_TBB = [lib]
    self.env.LIB_TBB = ['tbb']
    self.env.INCLUDES_TBB = [incl]
    self.env.DEFINES_TBB = ['USE_TBB']
    if using_oneapi:
        self.env.DEFINES_TBB += ['USE_TBB_ONEAPI']
