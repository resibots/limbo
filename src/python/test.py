import numpy as np
import limbo
from timeit import default_timer as timer

train_x = [np.random.randn(1) for i in range(0, 150)]
train_y = [np.sin(train_x[i]) + np.random.randn(1) * 0.1 for i in range(0, 150)]

print("init & opt GP...")
gp = limbo.make_gp(train_x, train_y, optimize_noise=True, iterations=300, verbose=True)
print("log likelihood:", gp.get_log_lik())

print('starting queries (1000)')
start = timer()
for i in range(0, 1000):
    x = np.random.randn(6)
    m, sigma = gp.query(x)

end = timer()
print('time for eval:', end-start)
print('', '')

print("init & opt Multi GP...")
train_x = [np.random.randn(2) for i in range(0, 150)]
train_y = [np.sin(train_x[i]) + np.random.randn(2) * 0.1 for i in range(0, 150)]
gp = limbo.make_multi_gp(train_x, train_y, optimize_noise=True, iterations=50, verbose=True)

print('starting queries (1000)')
start = timer()
for i in range(0, 1000):
    x = np.random.randn(6)
    m, sigma = gp.query(x)

end = timer()
print('time for eval:', end-start)

