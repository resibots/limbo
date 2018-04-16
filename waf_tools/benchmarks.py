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
import os, glob
import stat
import subprocess
import time
import threading
from waflib import Logs

plotting_ok = True

try:
    import plot_bo_benchmarks
    import plot_regression_benchmarks
except:
    plotting_ok = False
    Logs.pprint('YELLOW', 'YELLOW: Could not import plot_bo_benchmarks! Will not plot anything!')

json_ok = True
try:
    import simplejson
except:
    json_ok = False
    Logs.pprint('YELLOW', 'WARNING: simplejson not found some function may not work')

def run_bo_benchmarks(ctx):
    HEADER='\033[95m'
    NC='\033[0m'
    res_dir=os.getcwd()+"/benchmark_results/"
    try:
        os.makedirs(res_dir)
    except:
        Logs.pprint('YELLOW', 'WARNING: directory \'%s\' could not be created!' % res_dir)
    for fullname in glob.glob('build/src/benchmarks/*/*'):
        if os.path.isfile(fullname) and os.access(fullname, os.X_OK):
            fpath, fname = os.path.split(fullname)
            fpath, dir_name = os.path.split(fpath)
            directory = res_dir + "/" + dir_name + '/' + fname
            try:
                os.makedirs(directory)
            except:
                Logs.pprint('YELLOW', 'WARNING: directory \'%s\' could not be created, the new results will be concatenated to the old ones' % directory)
            s = "cp " + fullname + " " + directory
            retcode = subprocess.call(s, shell=True, env=None)
            if ctx.options.nb_rep:
                nb_rep = ctx.options.nb_rep
            else:
                nb_rep = 10
            for i in range(0,nb_rep):
                Logs.pprint('NORMAL', '%s Running: %s for the %s th time %s' % (HEADER, fname, str(i), NC))
                s="cd " + directory +";./" + fname
                retcode = subprocess.call(s, shell=True, env=None)
    # plot all if possible
    if plotting_ok:
        plot_bo_benchmarks.plot_all()


def compile_regression_benchmarks(bld, json_file):
    if not json_ok:
        Logs.pprint('RED', 'ERROR: simplejson is not installed and as such you cannot read the json configuration file for compiling the benchmarks.')
        return
    import types

    def convert(name):
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    configs = simplejson.load(open(json_file))
    for config in configs:
        name = config['name']
        funcs = config['functions']
        dims = config['dimensions']
        pts = config['points']
        # randomness = config['randomness']
        noise = config['noise']
        models = config['models']

        if len(models) < 1:
            Logs.pprint('YELLOW', 'ERROR: No model was found in the benchmark \'%s\'' % name)
            continue

        if len(dims) != len(funcs):
            dims = [dims]*len(funcs)
        if len(pts) != len(funcs):
            pts = [pts]*len(funcs)

        cpp_tpl = ""
        for line in open("waf_tools/benchmark_template.cpp"):
            cpp_tpl += line

        cpp_body = ""
        for i in range(len(funcs)):
            func = funcs[i]
            dim = dims[i]
            points = pts[i]

            dim_init = ', '.join(str(d) for d in dim)

            pts_init = ', '.join(str(p) for p in points)

            code = "    benchmark<" + func + ">(\"" + func.lower() + "\", {" + dim_init + "}, {" + pts_init + "});\n"
            cpp_body += code

        params_code = ""
        gps_learn_code = ""
        gps_query_code = ""

        model_names = []

        for m in range(len(models)):
            model = models[m]
            gp_type = 'GP'
            model_name = 'GP'
            kernel_type = 'SquaredExpARD'
            optimize_noise = 'true'
            kernel_params = {}
            mean_type = 'NullFunction'
            mean_has_defaults = True
            mean_params = {}
            hp_opt = 'KernelLFOpt'
            optimizer = 'Rprop'
            optimizer_params = []

            if 'name' in model:
                model_name = model['name']
            model_names.append(model_name)
            if 'type' in model:
                gp_type = model['type']
            if 'kernel' in model:
                kernel_type = model['kernel']['type']
                if 'optimize_noise' in model['kernel']:
                    optimize_noise = (model['kernel']['optimize_noise']).lower()
                if 'params' in model['kernel']:
                    kernel_params = model['kernel']['params']
            if 'mean' in model:
                mean_type = model['mean']['type']
                if 'has_defaults' in model['mean']:
                    mean_has_defaults = eval((model['mean']['has_defaults']).lower().title())
                if 'params' in model['mean']:
                    mean_params = model['mean']['params']
            if 'hp_opt' in model:
                hp_opt = model['hp_opt']['type']
                optimizer = model['hp_opt']['optimizer']
                if 'params' in model['hp_opt']:
                    optimizer_params = model['hp_opt']['params']

            if not isinstance(optimizer, types.ListType):
                optimizer = [optimizer]
                if len(optimizer_params) > 0:
                    optimizer_params = [optimizer_params]
            if not isinstance(optimizer_params, types.ListType):
                optimizer_params = [optimizer_params]

            kernel_find = kernel_type.rfind('::')
            mean_find = mean_type.rfind('::')

            kernel_underscore = convert(kernel_type[kernel_find if kernel_find>=0 else 0:])
            mean_underscore = convert(mean_type[mean_find if mean_find>=0 else 0:])

            if gp_type.find('::') == -1:
                gp_type = 'limbo::model::' + gp_type
            if kernel_find == -1:
                kernel_type = 'limbo::kernel::' + kernel_type
            if mean_find == -1:
                mean_type = 'limbo::mean::' + mean_type
            if hp_opt.find('::') == -1:
                hp_opt = 'limbo::model::gp::' + hp_opt

            params = "struct Params" + str(m) + " {\n    struct kernel {\n         BO_PARAM(double, noise, 0.01);\n         BO_PARAM(bool, optimize_noise, " + optimize_noise + ");\n    };\n"
            params += "\n    struct kernel_" + kernel_underscore + " : public defaults::kernel_" + kernel_underscore + " {\n"
            for key, value in kernel_params.items():
                params +=  "        BO_PARAM(" + value[0] + ", " + key + ", " + str(value[1]) + ");\n"
            params += "    };\n"
            mean_defaults = ""
            if mean_has_defaults == True:
                mean_defaults = "public defaults::mean_" + mean_underscore
            params += "\n    struct mean_" + mean_underscore + mean_defaults + " {\n"
            for key, value in mean_params.items():
                params +=  "        BO_PARAM(" + value[0] + ", " + key + ", " + str(value[1]) + ");\n"
            params += "    };\n"

            NLOptIds = []
            for o in range(len(optimizer)):
                if (o-1) in NLOptIds:
                    continue
                opt = optimizer[o]
                opt_find = opt.rfind('::')
                opt_underscore = convert(opt[opt_find if opt_find>=0 else 0:])
                # TO-DO: Maybe fix it in the code
                if opt == "ParallelRepeater":
                    opt_underscore = "parallelrepeater"
                if opt == "NLOptGrad":
                    opt_underscore = "nloptgrad"
                if opt == "NLOptNoGrad":
                    opt_underscore = "nloptnograd"
                if opt_find == -1:
                    opt = 'limbo::opt::' + opt
                params += "\n    struct opt_" + opt_underscore + " : public defaults::opt_" + opt_underscore + " {\n"
                if len(optimizer_params) > o:
                    for key, value in optimizer_params[o].items():
                        params +=  "        BO_PARAM(" + value[0] + ", " + key + ", " + str(value[1]) + ");\n"
                params += "    };\n"
                if optimizer[o] == "NLOptGrad" or optimizer[o] == "NLOptNoGrad":
                    NLOptIds.append(o)
            params += "};"

            tab_str = ''
            if m == 0:
                tab_str = '            '
            gp_code =  'using GP_' + str(m) + '_t = ' + gp_type+'<Params'+str(m)+', '+kernel_type+'<Params'+str(m)+'>, '+mean_type+'<Params'+str(m)+'>, '+hp_opt+'<Params'+str(m)+', '
            opt = optimizer[0]
            opt_find = opt.rfind('::')
            if opt_find == -1:
                opt = 'limbo::opt::' + opt
            gp_code += opt + '<Params'+str(m)+''
            for o in range(1, len(optimizer)):
                opt = optimizer[o]
                if (o-1) not in NLOptIds:
                    opt_find = opt.rfind('::')
                    if opt_find == -1:
                        opt = 'limbo::opt::' + opt
                    gp_code += ', ' + opt + '<Params'+str(m)+'>'
                else:
                    gp_code += ', ' + opt + '>'
            if 0 not in NLOptIds:
                gp_code += '>'
            gp_code += '>>;\n'
            gp_code += '            GP_' + str(m) + '_t gp_' + str(m) + ';\n\n'
            gp_code += '            auto start_' + str(m) + ' = std::chrono::high_resolution_clock::now();\n'
            gp_code += '            gp_' + str(m) + '.compute(points, obs, false);\n'
            gp_code += '            gp_' + str(m) + '.optimize_hyperparams();\n'
            gp_code += '            auto time_' + str(m) + ' = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_' + str(m) + ').count();\n'
            gp_code += '            std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);\n'
            gp_code += '            std::cout << "Time_' + str(m) + ' in secs: " << time_' + str(m) + '/ double(1000000.0) << std::endl;\n'

            gp_query_code  = '            std::vector<Eigen::VectorXd> predict_' + str(m) + '(N_test);\n'
            gp_query_code += '            start_' + str(m) + ' = std::chrono::high_resolution_clock::now();\n'
            gp_query_code += '            for (int i = 0; i < N_test; i++) {\n'
            gp_query_code += '                double ss;\n'
            gp_query_code += '                std::tie(predict_' + str(m) +'[i], ss) = gp_' + str(m) + '.query(test_points[i]);\n'
            gp_query_code += '            }\n'
            gp_query_code += '            auto time2_' + str(m) + ' = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_' + str(m) + ').count();\n'
            gp_query_code += '            std::cout << "Time_' + str(m) + '(query) in ms: " << time2_' + str(m) + ' * 1e-3 / double(N_test) << std::endl;\n\n'
            gp_query_code += '            double err_' + str(m) + ' = 0.0;\n'
            gp_query_code += '            for (int i = 0; i < N_test; i++) {\n'
            gp_query_code += '                err_' + str(m) + ' += (predict_' + str(m) +'[i] - test_obs[i]).squaredNorm();\n'
            gp_query_code += '            }\n'
            gp_query_code += '            err_' + str(m) + ' /= double(N_test);\n'
            gp_query_code += '            std::cout << "MSE(' + str(m) + '): " << err_' + str(m) + ' << std::endl;\n'

            params_code += params + '\n'
            gps_learn_code += gp_code
            gps_query_code += gp_query_code

        gps_query_code += '            ofs_res << std::setprecision(std::numeric_limits<long double>::digits10 + 1);\n'
        for m in range(len(models)):
            gps_query_code += '            ofs_res << time_' + str(m) + ' / double(1000000.0) << \" \" << time2_' + str(m) + ' * 1e-3 / double(N_test) << \" \" << err_' + str(m) + ' << " ' + model_names[m] + '" << std::endl;\n'

        cpp_tpl = cpp_tpl.replace('@NMODELS', str(len(models)))
        cpp_tpl = cpp_tpl.replace('@FUNCS', cpp_body)
        cpp_tpl = cpp_tpl.replace('@NOISE', noise)
        cpp_tpl = cpp_tpl.replace('@PARAMS', params_code)
        cpp_tpl = cpp_tpl.replace('@GPSLEARN', gps_learn_code)
        cpp_tpl = cpp_tpl.replace('@GPSQUERY', gps_query_code)

        if not os.path.exists(name + "_dir/"):
            os.makedirs(name + "_dir/")

        cpp = open(name + "_dir/" + name + ".cpp", "w")
        cpp.truncate()
        cpp.write(cpp_tpl)
        cpp.close()

        bld.program(features='cxx',
                    source=name + "_dir/" + name + ".cpp",
                    target=name,
                    includes='./src ./src/benchmarks/regression',
                    uselib='BOOST EIGEN TBB SFERES LIBCMAES NLOPT MKL_TBB LIBGP',
                    cxxflags=bld.env['CXXFLAGS'],
                    use='limbo')

def run_regression_benchmarks(ctx):
    HEADER='\033[95m'
    NC='\033[0m'

    if not json_ok:
        Logs.pprint('RED', 'ERROR: simplejson is not installed and as such you cannot read the json configuration file for running the benchmarks.')
        return

    if not ctx.options.regression_benchmarks:
        Logs.pprint('RED', 'ERROR: No json file with configurations is provided. Nothing will run!')
        return

    json_file = ctx.options.regression_benchmarks
    configs = simplejson.load(open(json_file))
    names = []
    for config in configs:
        names.append(config['name'])

    funcs = config['functions']
    dims = config['dimensions']
    pts = config['points']

    if len(dims) != len(funcs):
        dims = [dims]*len(funcs)
    if len(pts) != len(funcs):
        pts = [pts]*len(funcs)

    if ctx.options.nb_rep:
        nb_rep = ctx.options.nb_rep
    else:
        nb_rep = 5

    res_dir=os.getcwd()+"/regression_benchmark_results/"
    try:
        os.makedirs(res_dir)
    except:
        Logs.pprint('YELLOW', 'WARNING: directory \'%s\' could not be created!' % res_dir)
    for name in names:
        fullname = 'build/' + name
        if os.path.isfile(fullname) and os.access(fullname, os.X_OK):
            fpath, fname = os.path.split(fullname)
            directory = res_dir + "/" + fname

            # create directories first
            for i in range(0,nb_rep):
                exp_i = directory + "/exp_" + str(i)
                try:
                    os.makedirs(exp_i)
                except:
                    Logs.pprint('YELLOW', 'WARNING: directory \'%s\' could not be created' % exp_i)
                s = "cp " + fullname + " " + exp_i
                retcode = subprocess.call(s, shell=True, env=None)
                gpy_name = os.getcwd()+"/src/benchmarks/regression/gpy.py"
                s = "cp " + gpy_name + " " + exp_i
                retcode = subprocess.call(s, shell=True, env=None)

    for n in range(len(names)):
        name = names[n]
        directory = res_dir + "/" + name
        fullname = 'build/' + name
        # run experiments
        for i in range(0,nb_rep):
            Logs.pprint('NORMAL', '%s Running (limbo): %s for the %s-th time %s' % (HEADER, name, str(i), NC))
            exp_i = directory + "/exp_" + str(i)
            s="cd " + exp_i +";./" + name
            retcode = subprocess.call(s, shell=True, env=None)

        # run GPy experiments
        for i in range(0,nb_rep):
            Logs.pprint('NORMAL', '%s Running (GPy): %s for the %s-th time %s' % (HEADER, name, str(i), NC))
            exp_i = directory + "/exp_" + str(i)
            for k in range(len(funcs)):
                s="cd " + exp_i +"; python gpy.py " + funcs[k] + " '" + str(dims[k]) + "' '" + str(pts[k]) + "'"
                retcode = subprocess.call(s, shell=True, env=None)

    # plot all if possible
    if plotting_ok:
        plot_regression_benchmarks.plot_all()
