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
import os
import stat
import subprocess
import time
import threading
import params
import license
from waflib import Logs
from waflib.Tools import waf_unit_test

json_ok = True
try:
    import simplejson
except:
    json_ok = False
    Logs.pprint('YELLOW', 'WARNING: simplejson not found some function may not work')

def add_create_options(opt):
    opt.add_option('--dim_in', type='int', dest='dim_in', help='Number of dimensions for the function to optimize [default: 1]')
    opt.add_option('--dim_out', type='int', dest='dim_out', help='Number of dimensions for the function to optimize [default: 1]')
    opt.add_option('--bayes_opt_boptimizer_noise', type='float', dest='bayes_opt_boptimizer_noise', help='Acquisition noise of the function to optimize [default: 1e-6]')
    opt.add_option('--bayes_opt_bobase_stats_disabled', action='store_true', dest='bayes_opt_bobase_stats_disabled', help='Disable statistics [default: false]')
    opt.add_option('--init_randomsampling_samples', type='int', dest='init_randomsampling_samples', help='Number of samples used for the initialization [default: 10]')
    opt.add_option('--stop_maxiterations_iterations', type='int', dest='stop_maxiterations_iterations', help='Number of iterations performed before stopping the optimization [default: 190]')

# check if a lib exists for both osx (darwin) and GNU/linux
def check_lib(self, name, path):
    if self.env['DEST_OS']=='darwin':
        libname = name + '.dylib'
    else:
        libname = name + '.so'
    res = self.find_file(libname, path)
    lib = res[:-len(libname)-1]
    return res, lib

def create_variants(bld, source, uselib_local,
                    uselib, variants, includes=". ../",
                    cxxflags='',
                    target=''):
    source_list = source.split()
    if not target:
        tmp = source_list[0].replace('.cpp', '')
    else:
        tmp = target
    for v in variants:
        deff = []
        suff = ''
        for d in v.split(' '):
            suff += d.lower() + '_'
            deff.append(d)
        bin_fname = tmp + '_' + suff[0:len(suff) - 1]
        bld.program(features='cxx',
                    source=source,
                    target=bin_fname,
                    includes=includes,
                    uselib=uselib,
                    cxxflags=cxxflags,
                    use=uselib_local,
                    defines=deff)

def create_exp(name, opt):
    if not os.path.exists('exp'):
        os.makedirs('exp')
    if os.path.exists('exp/' + name):
        Logs.pprint('RED', 'ERROR: experiment \'%s\' already exists. Please remove it if you want to re-create it from scratch.' % name)
        return
    os.mkdir('exp/' + name)

    ws_tpl = ""
    for line in open("waf_tools/exp_template.wscript"):
        ws_tpl += line
    ws_tpl = ws_tpl.replace('@NAME', name)
    ws = open('exp/' + name + "/wscript", "w")
    ws.write(ws_tpl)
    ws.close()

    cpp_tpl = ""
    for line in open("waf_tools/exp_template.cpp"):
        cpp_tpl += line

    cpp_params = {}
    cpp_params['BAYES_OPT_BOPTIMIZER_NOISE'] = '    BO_PARAM(double, noise, ' + str(opt.bayes_opt_boptimizer_noise) + ');\n    ' if opt.bayes_opt_boptimizer_noise and opt.bayes_opt_boptimizer_noise >= 0 else ''
    cpp_params['BAYES_OPT_BOBASE_STATS_DISABLED'] = '    BO_PARAM(bool, stats_enabled, false);\n    ' if opt.bayes_opt_bobase_stats_disabled else ''
    cpp_params['INIT_RANDOMSAMPLING_SAMPLES'] = '    BO_PARAM(int, samples, ' + str(opt.init_randomsampling_samples) + ');\n    ' if opt.init_randomsampling_samples and opt.init_randomsampling_samples > 0  else ''
    cpp_params['STOP_MAXITERATIONS_ITERATIONS'] = '    BO_PARAM(int, iterations, ' + str(opt.stop_maxiterations_iterations) + ');\n    ' if opt.stop_maxiterations_iterations and opt.stop_maxiterations_iterations > 0 else ''

    cpp_params['DIM_IN'] = str(opt.dim_in) if opt.dim_in and opt.dim_in > 1 else '1'
    cpp_params['DIM_OUT'] = str(opt.dim_out) if opt.dim_out and opt.dim_out > 1 else '1'

    if opt.dim_in and opt.dim_in > 1:
        cpp_params['CODE_BEST_SAMPLE'] = 'boptimizer.best_sample().transpose()'
    else:
        cpp_params['CODE_BEST_SAMPLE'] = 'boptimizer.best_sample()(0)'

    if opt.dim_out and opt.dim_out > 1:
        cpp_params['CODE_RES_INIT'] = 'Eigen::VectorXd res(' + str(opt.dim_in) + ')'
        cpp_params['CODE_RES_RETURN'] = 'return res;'
        cpp_params['CODE_BEST_OBS'] = 'boptimizer.best_observation().transpose()'
    else:
        cpp_params['CODE_RES_INIT'] = 'double y = 0;'
        cpp_params['CODE_RES_RETURN'] = '// return a 1-dimensional vector\n        return tools::make_vector(y);'
        cpp_params['CODE_BEST_OBS'] = 'boptimizer.best_observation()(0)'

    for key, value in cpp_params.items():
        cpp_tpl = cpp_tpl.replace('@' + key, value)

    cpp = open('exp/' + name + "/" + name + ".cpp", "w")
    cpp.write(cpp_tpl)
    cpp.close()

def summary(bld):
    lst = getattr(bld, 'utest_results', [])
    total = 0
    tfail = 0
    if lst:
        total = len(lst)
        tfail = len([x for x in lst if x[1]])
    waf_unit_test.summary(bld)
    if tfail > 0:
        bld.fatal("Build failed, because some tests failed!")

def _sub_script(tpl, conf_file):
    if 'LD_LIBRARY_PATH' in os.environ:
        ld_lib_path = os.environ['LD_LIBRARY_PATH']
    else:
        ld_lib_path = "''"
    Logs.pprint('NORMAL', 'LD_LIBRARY_PATH=%s' % ld_lib_path)
     # parse conf
    list_exps = simplejson.load(open(conf_file))
    fnames = []
    for conf in list_exps:
        exps = conf['exps']
        nb_runs = conf['nb_runs']
        res_dir = conf['res_dir']
        bin_dir = conf['bin_dir']
        wall_time = conf['wall_time']
        use_mpi = "false"
        try:
            use_mpi = conf['use_mpi']
        except:
            use_mpi = "false"
        try:
            nb_cores = conf['nb_cores']
        except:
            nb_cores = 1
        try:
            args = conf['args']
        except:
            args = ''
        email = conf['email']
        if (use_mpi == "true"):
            ppn = '1'
            mpirun = 'mpirun'
        else:
     #      nb_cores = 1;
            ppn = "8"
            mpirun = ''

        #fnames = []
        for i in range(0, nb_runs):
            for e in exps:
                directory = res_dir + "/" + e + "/exp_" + str(i)
                try:
                    os.makedirs(directory)
                except:
                    Logs.pprint('YELLOW', 'WARNING: directory \'%s\' could not be created' % directory)
                subprocess.call('cp ' + bin_dir + '/' + e + ' ' + directory, shell=True)
                src_dir = bin_dir.replace('build/',  '')
                subprocess.call('cp ' + src_dir + '/params_*.txt '  + directory, shell=True)
                fname = directory + "/" + e + "_" + str(i) + ".job"
                f = open(fname, "w")
                f.write(tpl
                        .replace("@exp", e)
                        .replace("@email", email)
                        .replace("@ld_lib_path", ld_lib_path)
                        .replace("@wall_time", wall_time)
                        .replace("@dir", directory)
                        .replace("@nb_cores", str(nb_cores))
                        .replace("@ppn", ppn)
                        .replace("@exec", mpirun + ' ' + directory + '/' + e + ' ' + args))
                f.close()
                os.chmod(fname, stat.S_IEXEC | stat.S_IREAD | stat.S_IWRITE)
                fnames += [(fname, directory)]
    return fnames

def _sub_script_local(conf_file):
    if 'LD_LIBRARY_PATH' in os.environ:
        ld_lib_path = os.environ['LD_LIBRARY_PATH']
    else:
        ld_lib_path = "''"
    Logs.pprint('NORMAL', 'LD_LIBRARY_PATH=%s' % ld_lib_path)
     # parse conf
    list_exps = simplejson.load(open(conf_file))
    fnames = []
    for conf in list_exps:
        exps = conf['exps']
        nb_runs = conf['nb_runs']
        res_dir = conf['res_dir']
        bin_dir = conf['bin_dir']
        wall_time = conf['wall_time']
        use_mpi = "false"
        try:
            use_mpi = conf['use_mpi']
        except:
            use_mpi = "false"
        try:
            nb_cores = conf['nb_cores']
        except:
            nb_cores = 1
        try:
            args = conf['args']
        except:
            args = ''
        email = conf['email']
        if (use_mpi == "true"):
            ppn = '1'
            mpirun = 'mpirun'
        else:
     #      nb_cores = 1;
            ppn = "8"
            mpirun = ''

        #fnames = []
        for i in range(0, nb_runs):
            for e in exps:
                directory = res_dir + "/" + e + "/exp_" + str(i)
                try:
                    os.makedirs(directory)
                except:
                    Logs.pprint('YELLOW', 'WARNING: directory \'%s\' could not be created' % directory)
                subprocess.call('cp ' + bin_dir + '/' + e + ' ' + '"' + directory + '"', shell=True)
                src_dir = bin_dir.replace('build/',  '')
                subprocess.call('cp ' + src_dir + '/params_*.txt '  + directory, shell=True)
                fname = e
                fnames += [(fname, directory)]
    return fnames,args

def run_local_one(directory, s):
    std_out = open(directory+"/stdout.txt", "w")
    std_err = open(directory+"/stderr.txt", "w")
    retcode = subprocess.call(s, shell=True, env=None, stdout=std_out, stderr=std_err)

def run_local(conf_file, serial = True):
    if not json_ok:
        Logs.pprint('RED', 'ERROR: simplejson is not installed and as such you cannot read the json configuration file for running your experiments.')
        return
    fnames,arguments = _sub_script_local(conf_file)
    threads = []
    for (fname, directory) in fnames:
        s = "cd " + '"' + directory + '"' + " && " + "./" + fname + ' ' + arguments
        Logs.pprint('NORMAL', "Executing: %s" % s)
        if not serial:
            t = threading.Thread(target=run_local_one, args=(directory,s,))
            threads.append(t)
        else:
            run_local_one(directory,s)

    if not serial:
        for i in range(len(threads)):
            threads[i].start()
        for i in range(len(threads)):
            threads[i].join()

def qsub(conf_file):
    tpl = """#!/bin/sh
#? nom du job affiche
#PBS -N @exp
#PBS -o stdout
#PBS -b stderr
#PBS -M @email
# maximum execution time
#PBS -l walltime=@wall_time
# mail parameters
#PBS -m abe
# number of nodes
#PBS -l nodes=@nb_cores:ppn=@ppn
#PBS -l pmem=5200mb -l mem=5200mb
export LD_LIBRARY_PATH=@ld_lib_path
exec @exec
"""
    if not json_ok:
        Logs.pprint('RED', 'ERROR: simplejson is not installed and as such you cannot read the json configuration file for running your experiments.')
        return
    fnames = _sub_script(tpl, conf_file)
    for (fname, directory) in fnames:
        s = "qsub -d " + directory + " " + fname
        Logs.pprint('NORMAL', 'executing: %s' % s)
        retcode = subprocess.call(s, shell=True, env=None)
        Logs.pprint('NORMAL', 'qsub returned: %s' % str(retcode))


def oar(conf_file):
    tpl = """#!/bin/bash
#OAR -l /nodes=1/core=@nb_cores,walltime=@wall_time
#OAR -n @exp
#OAR -O stdout.%jobid%.log
#OAR -E stderr.%jobid%.log
export LD_LIBRARY_PATH=@ld_lib_path
exec @exec
"""
    if not json_ok:
        Logs.pprint('RED', 'ERROR: simplejson is not installed and as such you cannot read the json configuration file for running your experiments.')
        return
    Logs.pprint('YELLOW', 'WARNING [oar]: MPI not supported yet')
    fnames = _sub_script(tpl, conf_file)
    for (fname, directory) in fnames:
        s = "oarsub -d " + directory + " -S " + fname
        Logs.pprint('NORMAL', 'executing: %s' % s)
        retcode = subprocess.call(s, shell=True, env=None)
        Logs.pprint('NORMAL', 'oarsub returned: %s' % str(retcode))

def output_params(folder):
    files = [each for each in os.listdir(folder) if each.endswith('.cpp')]
    output = ''
    for file in files:
        output += 'FILE: ' + folder + '/' + file + '\n\n'
        output += params.get_output(folder + '/' + file)
        output += '=========================================\n'

    text_file = open(folder + "/params_" + folder[4:] + ".txt", "w")
    text_file.write(output)
    text_file.close()

def write_default_params(fname):
    output = "Default values\n"
    output += "===============\n"
    output += params.get_default_params()
    text_file = open(fname, "w")
    text_file.write(output)
    text_file.close()

def insert_license(): license.insert()
