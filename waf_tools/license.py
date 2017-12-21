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
# functions to insert the license in the headers of each cpp/hpp/py/wscript file
# note that we add a pipe (|) on each line so that we can remove the license and
# reinsert it automatically
import fnmatch,re
import os, shutil, sys

license= '''Copyright Inria May 2015
This project has received funding from the European Research Council (ERC) under
the European Union's Horizon 2020 research and innovation programme (grant
agreement No 637972) - see http://www.resibots.eu

Contributor(s):
  - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
  - Antoine Cully (antoinecully@gmail.com)
  - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
  - Federico Allocati (fede.allocati@gmail.com)
  - Vaios Papaspyros (b.papaspyros@gmail.com)
  - Roberto Rama (bertoski@gmail.com)

This software is a computer library whose purpose is to optimize continuous,
black-box functions. It mainly implements Gaussian processes and Bayesian
optimization.
Main repository: http://github.com/resibots/limbo
Documentation: http://www.resibots.eu/limbo

This software is governed by the CeCILL-C license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL-C
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL-C license and that you accept its terms.
'''

def make_dirlist(folder, extensions):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for ext in extensions:
            for filename in fnmatch.filter(filenames, '*' + ext):
                matches.append(os.path.join(root, filename))
    return matches

def insert_header(fname, prefix, license, kept_header = []):
    input = open(fname, 'r')
    ofname = '/tmp/' + fname.split('/')[-1]
    output = open(ofname, 'w')
    for line in kept_header:
        output.write(line + '\n')
    for line in license.split('\n'):
        if len(line)>0:
            output.write(prefix + ' ' + line + '\n')
        else:
            output.write(prefix + '\n')
    for line in input:
        header = len(filter(lambda x: x == line[0:len(x)], kept_header)) != 0
        if (line[0:len(prefix)] != prefix) and (not header):
            output.write(line)
    output.close()
    shutil.move(ofname, fname)

def insert():
    # cpp
    cpp =  make_dirlist('src', ['.hpp', '.cpp'])
    for i in cpp:
        insert_header(i, '//|', license)
    # py
    py = make_dirlist('waf_tools', ['.py'])
    py += make_dirlist('.', ['wscript'])
    for i in py:
        insert_header(i, '#|', license, ['#!/usr/bin/env python', '# encoding: utf-8'])
