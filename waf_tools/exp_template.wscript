#!/usr/bin/env python
def configure(conf):
    pass

def options(opt):
    pass

def build(bld):
    bld(features='cxx cxxprogram',
        source='@NAME.cpp',
        includes='. ../../src',
        target='@NAME',
        uselib='BOOST EIGEN TBB LIBCMAES NLOPT',
        use='limbo')
