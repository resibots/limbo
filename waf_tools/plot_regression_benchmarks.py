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
#|# plot the results of the Bayesian Optimization benchmarks
from glob import glob
from collections import defaultdict, OrderedDict
from datetime import datetime
import platform
import multiprocessing
import os

try:
    from waflib import Logs
    def print_log(c, s): Logs.pprint(c, s)
except: # not in waf
    def print_log(c, s): print(s)

try:
    import numpy as np
    from pylab import *
    import brewer2mpl
    bmap = brewer2mpl.get_map('Set2', 'qualitative', 8)
    colors = bmap.mpl_colors
    plot_ok = True
except:
    plot_ok = False
    Logs.pprint('YELLOW', 'WARNING: numpy/matplotlib not found: no plot of the BO benchmark results')

params = {
    'axes.labelsize' : 12,
    'text.fontsize' : 12,
    'axes.titlesize': 12,
    'legend.fontsize' : 12,
    'xtick.labelsize': 12,
    'ytick.labelsize' : 12,
    'figure.figsize' : [8, 8]
}
rcParams.update(params)

def load_data():
    files = glob("regression_benchmark_results/*/*/*.dat")
    points = defaultdict(lambda : defaultdict(lambda : defaultdict(dict)))
    times_learn = defaultdict(lambda : defaultdict(lambda : defaultdict(dict)))
    times_query = defaultdict(lambda : defaultdict(lambda : defaultdict(dict)))
    mses = defaultdict(lambda : defaultdict(lambda : defaultdict(dict)))
    for f in files:
        if 'test.dat' not in f and 'data.dat' not in f:
            fs = f.split("/")
            func, exp, bench = fs[-1][:-4], fs[-2], fs[-3]
            var = 'limbo'
            if func[-4:] == '_gpy':
                func = func[:-4]
                var = 'GPy'
            exp = exp[4:]

            text_file = open(f, "r")
            txt_d = text_file.readlines()
            n_models = int(txt_d[0].strip().split()[-1])
            var_base = var
            for i in range(0, len(txt_d), n_models+1):
                line = txt_d[i].strip().split()
                dim = int(line[0])
                pts = int(line[1])

                for j in range(0, n_models):
                    if n_models > 1:
                        var = var_base + '-' + str(j+1)
                    line2 = txt_d[i+j+1].strip().split()
                    time_learn = float(line2[0])
                    time_query = float(line2[1])
                    mse = float(line2[2])
                    if len(line2) == 4:
                        var = var_base + '-' + line2[-1]
                    # print(bench,var,func,dim)
                    if not (var in points[bench][func][dim]):
                        points[bench][func][dim][var] = []
                        times_learn[bench][func][dim][var] = []
                        times_query[bench][func][dim][var] = []
                        mses[bench][func][dim][var] = []

                    points[bench][func][dim][var].append(pts)
                    times_learn[bench][func][dim][var].append(time_learn)
                    times_query[bench][func][dim][var].append(time_query)
                    mses[bench][func][dim][var].append(mse)
    return points,times_learn,times_query,mses

def custom_ax(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_axisbelow(True)
    ax.grid(axis='x', color="0.9", linestyle='-')

def plot_ax(ax, data, points, labely, disp_legend=True, disp_xaxis=False):
    labels = []
    replicates = 0
    kk = 0

    # sort the variants
    sorted_keys = sorted(points.keys(), reverse=True)

    # for each variant
    for vv in range(len(sorted_keys)):
        var = sorted_keys[vv]
        labels.append(var)
        var_p = points[var]
        var_data = data[var]

        pp = {}

        for i in range(len(var_p)):
            if var_p[i] not in pp:
                pp[var_p[i]] = []
            pp[var_p[i]].append(var_data[i])
        
        pp = OrderedDict(sorted(pp.items()))

        x_axis = pp.keys()
        dd = pp.values()

        y_axis = []
        y_axis_75 = []
        y_axis_25 = []
        for i in range(len(dd)):
            replicates = len(dd[i])
            y_axis.append(np.median(dd[i]))
            y_axis_75.append(np.percentile(dd[i], 75))
            y_axis_25.append(np.percentile(dd[i], 25))

        c_kk = colors[kk%len(colors)]
        ax.plot(x_axis, y_axis, '-o', color=c_kk, linewidth=3, markersize=5)
        ax.fill_between(x_axis, y_axis_75, y_axis_25, color=c_kk, alpha=0.15, linewidth=2)
        kk = kk + 1
    
    if disp_legend:
        ax.legend(labels)
    if disp_xaxis:
        ax.set_xlabel('Number of points')
    ax.set_ylabel(labely)
    custom_ax(ax)

    return replicates

def plot_data(bench, func, dim, mses, query_times, learning_times, points, rst_file):
    name = func+'_'+str(dim)
    fig, ax = plt.subplots(3, sharex=True)

    replicates = plot_ax(ax[0], mses, points, 'Mean Squared Error')
    plot_ax(ax[1], query_times, points, 'Querying time in ms', False)
    plot_ax(ax[2], learning_times, points, 'Learning time in seconds', False, True)

    fig.tight_layout()
    fig.savefig('regression_benchmark_results/'+bench+'_figs/'+name+'.png')
    close()

    rst_file.write(func.title() + " in " + str(dim) + "D\n")
    rst_file.write("-----------------\n\n")
    rst_file.write(str(replicates) + " replicates \n\n")
    rst_file.write(".. figure:: fig_benchmarks/" + bench + "_figs/" + name + ".png\n\n")

def plot(points,times_learn,times_query,mses,rst_file):
    # for each benchmark configuration
    for bench in points.keys():
        fig_dir = os.getcwd() + '/regression_benchmark_results/'+bench+'_figs/'
        try:
            os.makedirs(fig_dir)
            print("created :" + fig_dir)
        except:
            print('WARNING: directory \'%s\' could not be created! (it probably exists already)' % fig_dir)
        # for each function
        for func in points[bench].keys():
            # for each dimension
            for dim in points[bench][func].keys():
                print('plotting for benchmark: ' + bench + ', the function: ' + func + ' for dimension: ' + str(dim))
                name = bench+'_'+func+'_'+str(dim)

                plot_data(bench, func, dim, mses[bench][func][dim], times_query[bench][func][dim], times_learn[bench][func][dim], points[bench][func][dim], rst_file)

def plot_all():
    if not plot_ok:
        print_log('YELLOW', "No plot")
        return

    rst_file = open("regression_benchmark_results/regression_benchmarks.rst", "w")
    rst_file.write("Gaussian process regression benchmarks\n")
    rst_file.write("======================================\n\n")
    date = "{:%B %d, %Y}".format(datetime.datetime.now())
    node = platform.node()
    rst_file.write("*" + date + "* -- " + node + " (" + str(multiprocessing.cpu_count()) + " cores)\n\n")

    rst_file.write("- We compare to GPy (https://github.com/SheffieldML/GPy) \n")
    rst_file.write("- Mean Squared Error: lower is better\n")
    rst_file.write("- Learning time: lower is better\n")
    rst_file.write("- Querying time (for 1 by 1 query points): lower is better\n\n")
    rst_file.write("- In each replicate, all the variants (see below for the variants available) are using exactly the same data\n\n")
    rst_file.write("- The data are uniformly sampled and some noise is added (according to the variance of the data).\n\n")

    rst_file.write("Naming convention\n")
    rst_file.write("------------------\n\n")

    rst_file.write("- GP-SE-Full-Rprop: Limbo with Squared Exponential kernel where the signal noise, signal variance and kernel lengthscales are optimized via Maximum Likelihood Estimation with the Rprop optimizer (default for limbo)\n")
    rst_file.write("- GP-SE-Full-SLSQP: Limbo with Squared Exponential kernel where the signal noise, signal variance and kernel lengthscales are optimized via Maximum Likelihood Estimation with the SLSQP optimizer (provided by NLOpt)\n")
    rst_file.write("- GP-SE-Rprop: Limbo with Squared Exponential kernel where the signal variance and kernel lengthscales are optimized via Maximum Likelihood Estimation with the Rprop optimizer (default for limbo) and where the signal noise is not optimized but set to a default value: 0.01\n")
    rst_file.write("- GPy: GPy with Squared Exponential kernel where the signal noise, signal variance and kernel lengthscales are optimized via Maximum Likelihood Estimation (with the L-BFGS-B optimizer --- `check scipy.optimize.fmin_l_bfgs_b= <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`_)\n\n")

    print('loading data...')
    points,times_learn,times_query,mses = load_data()
    print('data loaded')
    plot(points,times_learn,times_query,mses,rst_file)

if __name__ == "__main__":
    plot_all()