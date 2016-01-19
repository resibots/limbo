from pylab import *
import sys

sys.argv.pop(0)

for i in sys.argv:
    data = np.loadtxt(i)
    plot(data[:, 0], data[:, 1], 'o', label=i)

x_theory = []
y_theory = []
for t in np.arange(-2, 2, 0.01):
 x_theory += [exp(-(t-1/sqrt(2))**2-(t-1/sqrt(2))**2)]
 y_theory += [exp(-(t+1/sqrt(2))**2-(t+1/sqrt(2))**2)]

plot(x_theory, y_theory, label='theory')
legend(loc=4)
show()
