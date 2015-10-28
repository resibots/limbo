#!/usr/bin/env python
# encoding: utf-8

import subprocess
import os


def create(bld):
    kernels = ['Exp', 'MaternThreeHalfs', 'MaternFiveHalfs', 'SquaredExpARD']
    kernel_incompatibility = {}
    kernel_incompatibility['Exp'] = ['GPAuto', 'GPAutoMean']
    kernel_incompatibility['MaternThreeHalfs'] = ['GPAuto', 'GPAutoMean']
    kernel_incompatibility['MaternFiveHalfs'] = ['GPAuto', 'GPAutoMean']

    means = ['NullFunction', 'Constant', 'Data', 'FunctionARD']
    mean_additional_params = {}
    mean_additional_params['FunctionARD'] = ['MeanEval']
    mean_incompatibiliy = {}
    mean_incompatibiliy['NullFunction'] = ['GPAutoMean']
    mean_incompatibiliy['Constant'] = ['GPAutoMean']
    mean_incompatibiliy['Data'] = ['GPAutoMean']

    models = ['GP', 'GPAuto', 'GPAutoMean']
    acquisitions = ['UCB', 'GP_UCB']
    inner_optis = ['Random', 'ExhaustiveSearch', 'Cmaes']
    inits = ['NoInit', 'RandomSampling', 'RandomSamplingGrid', 'GridSampling']
    stats = ['Acquisitions']
    stops = ['MaxIterations', 'MaxPredictedValue']

    stats = 'typedef boost::fusion::vector<' + ', '.join(['stats::' + stat + '<Params>' for stat in stats]) + '> stats_t;\n'
    stops = '    typedef boost::fusion::vector<' + ', '.join(['stop_crit::' + stop + '<Params>' for stop in stops]) + '> stops_t;\n'

    src_path = bld.path.abspath() + '/combinations'
    if not os.path.exists(src_path):
        os.makedirs(src_path)

    with open(bld.path.abspath() + '/all_combinations_template.cpp', 'r') as f:
        template = f.read()

    bld.add_post_fun(lambda ctx: subprocess.call('rm -rf ' + src_path, shell=True))

    i = 0
    for kernel in kernels:
        for mean in means:
            for model in models:
                if (kernel in kernel_incompatibility and model in kernel_incompatibility[kernel]) or (mean in mean_incompatibiliy and model in mean_incompatibiliy[mean]):
                    continue
                for acqui in acquisitions:
                    for inner_opt in inner_optis:
                        for init in inits:
                            declarations = stats + stops
                            declarations = declarations + '    typedef kernel_fun::' + kernel + '<Params> kernel_' + str(i) + '_t;\n'
                            declarations = declarations + '    typedef mean_fun::' + mean + '<Params' + ('' if (not mean in mean_additional_params) else ',' + ', '.join(mean_additional_params[mean])) + '>' + ' mean_' + str(i) + '_t;\n'
                            declarations = declarations + '    typedef models::' + model + '<Params, kernel_' + str(i) + '_t, mean_' + str(i) + '_t> model_' + str(i) + '_t;\n'
                            declarations = declarations + '    typedef acqui_fun::' + acqui + '<Params, model_' + str(i) + '_t> acqui_' + str(i) + '_t;\n'
                            declarations = declarations + '    typedef inner_opt::' + inner_opt + '<Params> inner_opt_' + str(i) + '_t;\n'
                            declarations = declarations + '    typedef init_fun::' + init + '<Params> init_' + str(i) + '_t;\n'
                            declarations = declarations + '    bayes_opt::BOptimizer<Params, modelfun<model_' + str(i) + '_t>, acquifun<acqui_' + str(i) + '_t>, inneropt<inner_opt_' + str(i) + '_t>, initfun<init_' + str(i) + '_t>, statsfun<stats_t>, stopcrit<stops_t>> opt_' + str(i) + ';\n'
                            with open(bld.path.abspath() + '/combinations/combinations_' + str(i) + '.cpp', 'w') as f:
                                f.write(template.replace('@declarations', declarations).replace('@optimizer', 'opt_' + str(i)))
                            bld.program(features='cxx',
                                        source='/combinations/combinations_' + str(i) + '.cpp',
                                        includes='. .. ../../',
                                        target='/combinations/combinations_' + str(i),
                                        uselib='BOOST EIGEN TBB',
                                        use='limbo')
                            i = i + 1
