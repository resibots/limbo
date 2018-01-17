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
import fnmatch,re
import os
import sys
from collections import defaultdict

def make_dirlist(folder, extensions):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for ext in extensions:
            for filename in fnmatch.filter(filenames, '*'+ext):
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
    for line in f:
        stripped_line = line.strip()
        if len(stripped_line)>1 and stripped_line[0] == '/' and stripped_line[1] == '/':
            continue
        if '{' in line:
            level += [line];
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
        if 'BO_PARAM(' in line and not "#define" in line and not '//' in line:
            namespace, struct = extract_namespace(level)
            type, name, value = extract_param(line)
            d = extract_ifdef(ifdefs)
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
        s += " -> " + self.type + ' ' + self.name + ' = ' + self.value + ' [from ' + self.fname + ']' + str(self.ifdef)
        return s

def underline(k):
    out = k + '\n'
    s = ''
    for i in range(0, len(k)):
        s += '^'
    out += s + '\n'
    return out

def extract_defaults():
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
    return defaults


def get_default_params():
    output = ''
    defaults = extract_defaults()
    for k in defaults.keys():
        output += underline(k)
        for kk in defaults[k].keys():
            output += '- ' + str(defaults[k][kk].type) + ' ' + str(defaults[k][kk].name) + ' = ' + str(defaults[k][kk].value) + ' [defined in ' + str(defaults[k][kk].fname) + ']'
            if defaults[k][kk].ifdef:
                output += ' ' + str(defaults[k][kk].ifdef)
            output += '\n'
        output += '\n'
    return output

def get_output(file):
    get_default_params()
    output = ""
    defaults = extract_defaults()
    # find the params in the current file
    p, d = extract_params(file)
    struct_set = set()
    plist = defaultdict(dict)
    for i in p:
        struct_set.add(i.struct)
        plist[i.struct][i.name] = i
    for i in d:
        struct_set.add(i)
    for k in struct_set:
        output += underline(k)
        for kk in defaults[k].keys():
            if kk in plist[k]:
                output += '- ' + str(plist[k][kk].type) + ' ' + str(plist[k][kk].name) + ' = ' + str(plist[k][kk].value) + ' [defined in ' + plist[k][kk].fname + ']'  + str(plist[k][kk].ifdef) + '\n'
            else:
                output += '- ' + str(defaults[k][kk].type) + ' ' + str(defaults[k][kk].name) + ' = ' + str(defaults[k][kk].value) + ' [default value, from ' + str(defaults[k][kk].fname) + ']'  + str(defaults[k][kk].ifdef) + '\n'
        output += '\n'

    return output
