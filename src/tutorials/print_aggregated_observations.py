# this will load numpy for us
import matplotlib.pyplot as plt
import numpy as np
import sys

if(len(sys.argv)<2):
    print("ERROR : Please provide path to the aggregated_observations.dat file.")
    quit()
    
# Load and process data
data = np.loadtxt(sys.argv[1],skiprows=1)
x = data[:,0]
y = data[:,1]
min_y=np.min(y)
max_y=np.max(y)
rand_init_x = x[x==-1]
rand_init_y = y[x==-1]

y = y[x!=-1]
x = x[x!=-1]

# set vertical limit
plt.ylim(min_y-0.1*np.abs(max_y-min_y), max_y+0.1*np.abs(max_y-min_y))

# plot random initilization and observations
rand_init = plt.plot(rand_init_x, rand_init_y,'g^', label='Random Initialization')
observations = plt.plot(x, y,'bo',label = 'observations')
plt.xlabel('Iterations')
plt.ylabel('Observations')
plt.title('Limbo')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
