import GPy
import numpy as np
import time
import sys
import os.path

def benchmark(name, dims, points):
    # dir_prefix = 'regression_benchmarks/'+name
    dir_prefix = './'
    N_test = 10000
    Ns = points

    file_results = open(name+'_gpy.dat', 'w')
    file_results.truncate()

    for D in dims:
        for N in Ns:
            fname = dir_prefix+'/'+name+'_'+str(D)+'_'+str(N)+'_data.dat'
            if not os.path.isfile(fname):
                continue
            file_results.write(str(D) + " " + str(N) + " 1\n")
            # Load training data
            samples = np.zeros((N,D))
            observations = np.zeros((N,1))
            matrix_load = np.loadtxt(dir_prefix+'/'+name+'_'+str(D)+'_'+str(N)+'_data.dat')
            samples = matrix_load[:,0:D]
            observations = matrix_load[:,-1].reshape((N,1))

            # train GP
            kern = GPy.kern.RBF(samples.shape[1], ARD=True, inv_l=False)
            lik = GPy.likelihoods.Gaussian()

            start = time.time()
            m = GPy.core.GP(samples, observations, kernel=kern, likelihood=lik)
            m.optimize(messages=False, ipython_notebook=False)#'RProp', max_iters=300, messages=False, ipython_notebook=False)
            end = time.time()

            # Load test data
            test_points = np.zeros((N_test,D))
            test_obs = np.zeros((N_test,1))
            matrix_load = np.loadtxt(dir_prefix+'/'+name+'_'+str(D)+'_'+str(N)+'_test.dat')
            test_points = matrix_load[:,0:D]
            test_obs = matrix_load[:,-1].reshape((N_test,1))

            # Make predictions
            Y_gp = np.zeros((N_test,D))
            var_gp = np.zeros((N_test,1))

            start2 = time.time()
            # (Y_gp, var_gp) = m.predict(test_points)
            for i in range(N_test):
                (Y_gp[i,:], var_gp[i,:]) = m.predict(test_points[i,:].reshape((1,D)))
            end2 = time.time()

            # Compute MSE
            mse = 0.0
            for i in range(N_test):
                nn = np.linalg.norm(Y_gp[i,0]-test_obs[i,0])
                mse += nn*nn
            mse = mse/float(N_test)

            print name + ': ', D, N, mse
            print 'Time: ' + str((end-start))
            print 'Time (query): ' + str(((end2-start2)*1000.0)/float(N_test))
            file_results.write(str((end-start)) + " " + str(((end2-start2)*1000.0)/float(N_test)) + " " + str(mse) + "\n")

    file_results.close()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        exit(1)
    benchmark(sys.argv[1].lower(), eval(sys.argv[2]), eval(sys.argv[3]))
