#!/bin/python
# plot the results of the Bayesian Optimization benchmarks
from glob import glob
from collections import defaultdict
import numpy as np
from pylab import *


params = {
    'axes.labelsize' : 8,
    'text.fontsize' : 8,
    'axes.titlesize': 10,
    'legend.fontsize' : 10,
    'xtick.labelsize': 5,
    'ytick.labelsize' : 10,
    'figure.figsize' : [9, 2.5]
}
rcParams.update(params)

def load_data():
    files = glob("../benchmark_results/*/*/*.dat")
    data = defaultdict(lambda : defaultdict(dict))
    for f in files:
        fs = f.split("/")
        func, var, lib = fs[-1], fs[-2], fs[-3]
        data[func][lib][var] = np.loadtxt(f)
    return data

def custom_ax(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.grid(axis='x', color="0.9", linestyle='-')

# plot a single function
def plot(func_name, data):
    d = data[func_name]
    da_acc = []
    da_time = []
    labels = [] 
    for k in d.iterkeys():
        for k2 in d[k].iterkeys():
            da_acc.append(d[k][k2][:, 0])
            da_time.append(d[k][k2][:, 1] / 1000.0)
            labels.append(k + "/" + k2)
    fig = figure()
    fig.subplots_adjust(left=0.3)
    ax = fig.add_subplot(121)
    custom_ax(ax)
    ax.boxplot(da_acc, 0, 'rs', 0)
    ax.set_yticklabels(labels)
    ax.set_title("Accuracy")
    ax = fig.add_subplot(122)
    custom_ax(ax)
    ax.boxplot(da_time, 0, 'rs', 0)
    ax.set_yticklabels([])
    ax.set_title("Wall clock time")

    fig.savefig("../benchmark_results/" + func_name.split('.')[0] + ".png")    


def main():
    data = load_data()
    for k in data.keys():
        plot(k, data)

main()