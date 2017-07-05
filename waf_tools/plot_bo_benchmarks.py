#!/bin/python
# plot the results of the Bayesian Optimization benchmarks
from glob import glob
from collections import defaultdict
import numpy as np
from pylab import *
import brewer2mpl
bmap = brewer2mpl.get_map('Set2', 'qualitative', 8)

colors = bmap.mpl_colors

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
    files = glob("benchmark_results/*/*/*.dat")
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
    ax.set_axisbelow(True)
    ax.grid(axis='x', color="0.9", linestyle='-')

def custom_boxes(ax, bp):
    for i in range(len(bp['boxes'])):
        box = bp['boxes'][i]
        box.set_linewidth(0)
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
            boxCoords = zip(boxX,boxY)
            boxPolygon = Polygon(boxCoords, facecolor = colors[i % len(colors)], linewidth=0)
            ax.add_patch(boxPolygon)

        for i in range(0, len(bp['boxes'])):
            bp['boxes'][i].set_color(colors[i])
            # we have two whiskers!
            bp['whiskers'][i*2].set_color(colors[i])
            bp['whiskers'][i*2 + 1].set_color(colors[i])
            bp['whiskers'][i*2].set_linewidth(2)
            bp['whiskers'][i*2 + 1].set_linewidth(2)
            # top and bottom fliers
            bp['fliers'][i * 2].set(markerfacecolor=colors[i],
                            marker='o', alpha=0.75, markersize=6,
                            markeredgecolor='none')
            bp['fliers'][i * 2 + 1].set(markerfacecolor=colors[i],
                            marker='o', alpha=0.75, markersize=6,
                            markeredgecolor='none')
            bp['medians'][i].set_color('black')
            bp['medians'][i].set_linewidth(2)
            # and 4 caps to remove
            for c in bp['caps']:
                c.set_linewidth(0)


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
    bp = ax.boxplot(da_acc, 0, 'rs', 0)
    custom_boxes(ax, bp)
    ax.set_yticklabels(labels)
    ax.set_title("Accuracy")
    ax = fig.add_subplot(122)
    custom_ax(ax)
    bp = ax.boxplot(da_time, 0, 'rs', 0)
    custom_boxes(ax, bp)
    ax.set_yticklabels([])
    ax.set_title("Wall clock time")

    fig.savefig("benchmark_results/" + func_name.split('.')[0] + ".png")    


def plot_all():
    print('loading data...')
    data = load_data()
    print('data loaded')
    for k in data.keys():
        print('plotting for ' + k + '...')
        plot(k, data)

if __name__ == "__main__":
    plot_all()