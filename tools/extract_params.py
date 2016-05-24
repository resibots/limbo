#!/usr/bin/env python
import fnmatch
import re
import os
import sys
from collections import defaultdict


def make_dirlist(folder, extensions):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for ext in extensions:
            for filename in fnmatch.filter(filenames, '*' + ext):
                matches.append(os.path.join(root, filename))
    return matches


def extract_namespace(level):
    namespace = []
    struct = ''
    for i in level:
        if 'namespace' in i:
            d = re.findall('namespace [a-zA-Z_]+', i)
            n = d[0].split(' ')[1]
            namespace += [n]
        if 'struct' in i:
            d = re.findall('struct [a-zA-Z_]+', i)
            n = d[0].split(' ')[1]
            struct = n
    return namespace, struct


def extract_param(line):
    i = line.replace(' ', '').replace('\t', '')
    k = re.split('(,|\(|\))', i)
    type = k[2]
    name = k[4]
    value = k[6]
    return type, name, value


def extract_ifdef(d):
    x = filter(lambda x: not ('_HPP' in x), d)
    x = map(lambda x: x.replace('NOT #ifndef', '#ifdef'), x)
    x = map(lambda x: x.replace('\n', ''), x)

    return x

def extract_params(fname):
    params = []
    f = open(fname)
    level = []
    defaults = []
    ifdefs = []
    elifdef = []
    for line in f:
        if '{' in line:
            level += [line]
        if '}' in line:
            level.pop(-1)
        if '#if' in line:
			ifdefs += [line]
        if '#else' in line:
            ifdefs[-1] = 'NOT ' + ifdefs[-1]
        if '#elif' in line:
            ifdefs[-1] = line
        if '#endif' in line:
            ifdefs.pop(-1)
        if 'defaults::' in line:
            d = re.findall('defaults::\w+', line)[0]
            dd = d.split('::')[1]
            defaults += [dd]
        if 'BO_PARAM(' in line and not "#define" in line:
            namespace, struct = extract_namespace(level)
            d = extract_ifdef(ifdefs)
            type, name, value = extract_param(line)
            p = Param(namespace, struct, type, name, value, fname, d)
            params += [p]
    return params, defaults


class Param:

    def __init__(self, namespace, struct, type, name, value, fname, ifdef):
        self.namespace = namespace
        self.struct = struct
        self.type = type
        self.name = name
        self.value = value
        self.fname = fname
        self.ifdef = ifdef

    def to_str(self):
        s = ''
        for k in self.namespace:
            s += k + ' :: '
        s += self.struct
        s += " -> " + self.type + ' ' + self.name + ' = ' + \
            self.value + ' [from ' + self.fname + ']'
        print s


def underline(k):
    print k
    s = ''
    for i in range(0, len(k)):
        s += '='
    print s

if __name__ == "__main__":
    # find the default params
    dirs = make_dirlist('src/', ['.hpp'])
    params = []
    for fname in dirs:
        p, d = extract_params(fname)
        params += p
    defaults = defaultdict(dict)
    for i in params:
        if 'defaults' in i.namespace:
            defaults[i.struct][i.name] = i

    # if we have a filename
    if (len(sys.argv) > 1):
        # find the params in the current file
        p, d = extract_params(sys.argv[1])
        struct_set = set()
        plist = defaultdict(dict)
        for i in p:
            struct_set.add(i.struct)
            plist[i.struct][i.name] = i
        for i in d:
            struct_set.add(i)
        for k in struct_set:
            underline(k)
            for kk in defaults[k].keys():
                if kk in plist[k]:
                    print '-', plist[k][kk].type, plist[k][kk].name, '=', plist[k][kk].value, '[defined in ' + plist[k][kk].fname + ']', plist[k][kk].ifdef
                else:
                    print '-', defaults[k][kk].type, defaults[k][kk].name, '=', defaults[k][kk].value, '[default value, from ' + defaults[k][kk].fname + ']', plist[k][kk].ifdef
            print ''
    else:  # no filename, print the defaults in rst (for the doc)
        print "Default values"
        print '------------------------'
        print ''
        print '.. highlight:: c++'
        print ''
        for k in sorted(defaults.keys()):
            underline(k)
            print ''
            for kk in sorted(defaults[k].keys()):
                print '- ``' + defaults[k][kk].type + ' ' + defaults[k][kk].name + ' = ' + defaults[k][kk].value + '`` [default value, from ' + defaults[k][kk].fname + ']', defaults[k][kk].ifdef
            print '\n'
