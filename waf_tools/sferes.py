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
# JB Mouret - 2009

"""
Quick n dirty sferes2 detection
"""

from waflib.Configure import conf


def options(opt):
    opt.add_option('--sferes', type='string', help='path to sferes2', dest='sferes')


@conf
def check_sferes(self, *k, **kw):
    if self.options.sferes:
        includes_sferes = [self.options.sferes]
        libpath_sferes = [self.options.sferes + '/build/sferes']
    else:
        return

    self.start_msg('Checking sferes includes')
    try:
        self.find_file('sferes/ea/ea.hpp', includes_sferes)
        self.end_msg(True)
    except:
        self.end_msg(False)
        return

    self.start_msg('Checking sferes libs')
    try:
        self.find_file('libsferes2.a', libpath_sferes)
        self.end_msg(True)
    except:
        self.end_msg(False)
        return

    self.env.STLIBPATH_SFERES = libpath_sferes
    self.env.STLIB_SFERES = ["sferes2"]
    self.env.INCLUDES_SFERES = includes_sferes
    self.env.DEFINES_SFERES = ["USE_SFERES", "SFERES_FAST_DOMSORT"]
