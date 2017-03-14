import glob
from pylab import *
import brewer2mpl

 # brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Paired', 'qualitative', 12)
colors = bmap.mpl_colors

params = {
    'axes.labelsize': 8,
    'text.fontsize': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [10, 10]
}
rcParams.update(params)


points = np.loadtxt('out.dat')
num = np.loadtxt('num.dat')
splits = np.loadtxt('split.dat')
# bounds = np.loadtxt('bounds.dat')

fig = figure() # no frame
ax = fig.add_subplot(111)

px = []
py = []

N = len(num)

j = 0
for i in range(N):
    px.append([])
    py.append([])
    for k in range(int(num[i])):
        px[i].append(points[j+k][0])
        py[i].append(points[j+k][1])

    j = j + int(num[i])

# sp = splits
# pp = []
#
# for i in range(len(points)):
#     s = np.array([sp[0], sp[1]])
#     p = np.array([points[i][0], points[i][1]])
#     pp.append(s.dot(p))
#
# med = np.median(pp)
#
# j = 0
# for i in range(len(num)):
#     for k in range(int(num[i])):
#         s = np.array([sp[0], sp[1]])
#         p = np.array([points[j+k][0], points[j+k][1]])
#         v = s.dot(p)
#         if i == 0 and v > med:
#             print 'Error'
#         if i == 1 and v<= med:
#             print 'Error'
#     j = j + int(num[i])

for i in range(N):
    ax.plot(px[i], py[i], 'o', linewidth=2, color=colors[i%len(colors)], alpha=0.75, label=str(i+1))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

# # kk = 0
# for i in range(splits.shape[0]):
# # for i in range(kk, kk+5):
#     if len(splits.shape)==1:
#         sp = splits
#         # bd = bounds
#     else:
#         sp = splits[i,:]
#         # bd = bounds[i,:]
#     d = [-sp[1], sp[0]]
#     # min_x = bd[2]#np.min(px[i*2]+px[i*2-1])/2.0
#     # max_x = bd[4]#np.max(px[i*2]+px[i*2-1])/2.0
#     # min_y = bd[3]#np.min(py[i*2]+py[i*2-1])/2.0
#     # max_y = bd[5]#np.max(py[i*2]+py[i*2-1])/2.0
#     c = [sp[2], sp[3]]
#     # cv = [bd[0], bd[1]]
#     # cv = [sp[2], sp[3]]
#     # sp_dir_x = [c[0]-(c[0]-min_x)*d[0], c[0]-(c[0]-max_x)*d[0]]
#     # sp_dir_y = [c[1]-(c[1]-min_y)*d[1], c[1]-(c[1]-max_y)*d[1]]
#
#     min_x = sp[5]
#     max_x = sp[7]
#     min_y = sp[6]
#     max_y = sp[8]
#
#     x_b = np.abs(max_x - c[0])
#     x_s = np.abs(min_x - c[0])
#     # print x_s, x_b
#
#     y_b = np.abs(max_y - c[1])
#     y_s = np.abs(min_y - c[1])
#     # print y_s, y_b
#     y_space = np.linalg.norm(np.array([min_x, min_y]) - np.array([c[0], c[1]]))
#     x_space = np.linalg.norm(np.array([max_x, max_y]) - np.array([c[0], c[1]]))
#
#     # x_space = x_space*x_space
#     # y_space = y_space*y_space
#
#     sp_dir_x = [c[0]-y_space*d[0], c[0]+x_space*d[0]]
#     sp_dir_y = [c[1]-y_space*d[1], c[1]+x_space*d[1]]
#
#     # print sp_dir_x
#     # print sp_dir_y
#
#     # sp_dir_x = [c[0]-2*d[0], c[0]+2*d[0]]
#     # sp_dir_y = [c[1]-2*d[1], c[1]+2*d[1]]
#
#     # x_b = np.abs(max_x - cv[0])
#     # x_s = np.abs(min_x - cv[0])
#     # x_space = x_b
#     # if x_s>x_space:
#     #     x_space = x_s
#     #
#     # y_b = np.abs(max_y - cv[1])
#     # y_s = np.abs(min_y - cv[1])
#     #
#     # y_space = y_b
#     # if y_s>y_space:
#     #     y_space = y_s
#     #
#     # # print x_s, x_b, y_s, y_b
#     # # print c[0], c[1]
#     # # print c[0]-10*d[0], c[0]+10*d[0], c[1]-10*d[1], c[1]+10*d[1]
#     #
#     # # print sp_dir_x, sp_dir_y
#     # sp_dir_x = [cv[0]-x_s*d[0], cv[0]+x_b*d[0]]
#     # sp_dir_y = [cv[1]-y_s*d[1], cv[1]+y_b*d[1]]
#
#     # sp_dir_x2 = [cv[0]-x_s*d[0], cv[0]+x_b*d[0]]
#     # sp_dir_y2 = [cv[1]-y_s*d[1], cv[1]+y_b*d[1]]
#     # print sp_dir_x2, sp_dir_y2
#
#     # ax.plot(sp_dir_x, sp_dir_y, linewidth=3, color=colors[int(sp[4])])
#     # ax.plot(c[0], c[1], 'o', markersize=10, color=colors[int(sp[4])])
#     dd = [-sp[1], sp[0]]
#     ax.plot([c[0], c[0]+dd[0]*10], [c[1], c[1]+dd[1]*10], linewidth=5, color=colors[int(sp[4])])
#     ax.plot(c[0], c[1], 'o', markersize=10, color=colors[int(sp[4])])
#
#     # ax.plot([c[0], c[0]+d[0]], [c[1], c[1]+d[1]], '--', linewidth=5, color=colors[int(sp[4])])
#     # ax.plot(sp_dir_x2, sp_dir_y2, linewidth=8, color=colors[i])#color=colors[int(sp[4])])
#     # ax.plot(bd[0], bd[1], '*', markersize=20, color=colors[i])

xlim([-40,40])
ylim([-40,40])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='x', direction='out')
ax.tick_params(axis='y', length=0)
for spine in ax.spines.values():
  spine.set_position(('outward', 5))
ax.set_axisbelow(True)
ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)

fig.savefig('spatial.png')
