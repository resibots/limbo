#!/usr/bin/env python
# encoding: utf-8

import subprocess
import os


def create(bld):
    kernels = ['Exp', 'MaternThreeHalves', 'MaternFiveHalves', 'SquaredExpARD']
    kernel_incompatibility = {}
    kernel_incompatibility['Exp'] = ['KernelLFOpt', 'KernelMeanLFOpt', 'MeanLFOpt']
    kernel_incompatibility['MaternThreeHalves'] = ['KernelLFOpt', 'KernelMeanLFOpt', 'MeanLFOpt']
    kernel_incompatibility['MaternFiveHalves'] = ['KernelLFOpt', 'KernelMeanLFOpt', 'MeanLFOpt']

    means = ['NullFunction', 'Constant', 'Data', 'FunctionARD']
    mean_additional_params = {}
    mean_additional_params['FunctionARD'] = ['MeanEval']
    mean_incompatibiliy = {}
    mean_incompatibiliy['NullFunction'] = ['KernelMeanLFOpt', 'MeanLFOpt']
    mean_incompatibiliy['Constant'] = ['KernelMeanLFOpt', 'MeanLFOpt']
    mean_incompatibiliy['Data'] = ['KernelMeanLFOpt', 'MeanLFOpt']

    models = ['GP']
    gp_lf_optimizations = ['NoLFOpt', 'KernelLFOpt', 'KernelMeanLFOpt', 'MeanLFOpt']
    acquisitions = ['UCB', 'GP_UCB']
    optimizers = ['RandomPoint', 'GridSearch']
    if bld.env.DEFINES_NLOPT:
        optimizers += ['NLOptNoGrad']
    if bld.env.DEFINES_LIBCMAES:
        optimizers += ['Cmaes']

    inits = ['NoInit', 'RandomSampling', 'RandomSamplingGrid', 'GridSampling']
    stats = ['Samples', 'Observations', 'AggregatedObservations', 'BestSamples', 'BestObservations', 'BestAggregatedObservations',
             'GPPredictionDifferences', 'GPAcquisitions', 'GPLikelihood']

    stops = ['MaxIterations', 'MaxPredictedValue']

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
                for gp_lf_opt in gp_lf_optimizations:
                    if (kernel in kernel_incompatibility and gp_lf_opt in kernel_incompatibility[kernel]) or (mean in mean_incompatibiliy and gp_lf_opt in mean_incompatibiliy[mean]):
                        continue
                    for acqui in acquisitions:
                        for acqui_opt in optimizers:
                            for init in inits:
                                additional_stats = []
                                if kernel == 'SquaredExpARD':
                                    additional_stats.append('GPKernelHParams')
                                if mean == 'FunctionARD':
                                    additional_stats.append('GPMeanHParams')
                                declarations = 'using stats_t = boost::fusion::vector<' + ', '.join(['stat::' + stat + '<Params>' for stat in stats + additional_stats]) + '>;\n'
                                declarations = declarations + '    using stops_t = boost::fusion::vector<' + ', '.join(['stop::' + stop + '<Params>' for stop in stops]) + '>;\n'
                                declarations = declarations + '    using kernel_' + str(i) + '_t = kernel::' + kernel + '<Params>;\n'
                                declarations = declarations + '    using  mean_' + str(i) + '_t = mean::' + mean + '<Params' + ('' if (not mean in mean_additional_params) else ',' + ', '.join(mean_additional_params[mean])) + '>;\n'
                                declarations = declarations + '    using gp_lf_opt_' + str(i) + '_t = model::gp::' + gp_lf_opt + '<Params>;\n'
                                declarations = declarations + '    using model_' + str(i) + '_t = model::' + model + '<Params, kernel_' + str(i) + '_t, mean_' + str(i) + '_t, gp_lf_opt_' + str(i) + '_t>;\n'
                                declarations = declarations + '    using acqui_' + str(i) + '_t = acqui::' + acqui + '<Params, model_' + str(i) + '_t>;\n'
                                declarations = declarations + '    using acqui_opt_' + str(i) + '_t = opt::' + acqui_opt + '<Params>;\n'
                                declarations = declarations + '    using init_' + str(i) + '_t = init::' + init + '<Params>;\n'
                                declarations = declarations + '    bayes_opt::BOptimizer<Params, modelfun<model_' + str(i) + '_t>, acquifun<acqui_' + str(i) + '_t>, acquiopt<acqui_opt_' + str(i) + '_t>, initfun<init_' + str(i) + '_t>, statsfun<stats_t>, stopcrit<stops_t>> opt_' + str(i) + ';\n'
                                with open(bld.path.abspath() + '/combinations/combinations_' + str(i) + '.cpp', 'w') as f:
                                    f.write(template.replace('@declarations', declarations).replace('@optimizer', 'opt_' + str(i)))
                                bld.program(features='cxx',
                                            source='/combinations/combinations_' + str(i) + '.cpp',
                                            includes='. .. ../../',
                                            target='/combinations/combinations_' + str(i),
                                            uselib='BOOST EIGEN TBB SFERES LIBCMAES NLOPT',
                                            use='limbo')
                                i = i + 1
