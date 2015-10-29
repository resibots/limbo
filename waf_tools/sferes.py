#! /usr/bin/env python
# encoding: utf-8
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
        libpath_sferes = [self.options.sferes + '/build/default/sferes']
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
