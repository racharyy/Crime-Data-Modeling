import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import bernoulli
from scipy.sparse import rand
from ar_synthetic_generate import *
from RMLE_estimation import *
import csv


num_col = 10
data_dimension = num_col
issynthetic =False
nu = np.zeros(data_dimension)
sparsity =5
max_iter=10000
method='acc_prox_gd'


data=np.loadtxt('../../Data/AR_data/hotspot_crime_count.csv',delimiter=',',skiprows=1,usecols=range(1,11))
T=data.shape[0]
estimator = rmle(data,nu,data_dimension, sparsity,T,issynthetic)
y1,y2 = estimator.gd_run(max_iter=max_iter,method=method)
print len(y1),len(y2)


plt.plot(range(len(y2)), y2, 'k-')
plt.title('Evolution of Negative Log-Likelihood with iteration')
plt.xlabel('iteration')
plt.ylabel('Negative Log-Likelihood')
plt.savefig('../../Code/Autoregressive/Plots/Log-Likelihood_crime.png')
plt.show()





	

	






