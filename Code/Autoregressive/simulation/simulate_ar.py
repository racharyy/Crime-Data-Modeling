import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import bernoulli
from scipy.sparse import rand
from ar_synthetic_generate import *
from RMLE_estimation import *

data_dimension_ar = [5,10,20]

sparsity_ar = [5,10]
T_ar=[100,300,1000]
data =[]
max_iter=10000
method='acc_prox_gd'

#Pads extra elements to make uniform size array
def concat(ar,extend):
	if len(ar)<extend:
		ar = ar+list[ar[-1]*np.ones(extend-len(ar))]

	return ar


#Generate a synthetic data set and try to learn using Proximal Gradiant Descent num_run
#many times and plots the error bar for all the execution 
def test_on_synthetic(num_run,data,data_dimension,sparsity,T,issynthetic=True,model='Poisson'):

	nu = np.zeros(data_dimension)
	#Use RMLE to estimate the parameter matrix
	estimator = rmle(data,nu,data_dimension, sparsity,T,issynthetic,model)
	print estimator.data.shape
	mean_1,mean_2=[],[]
	max_1,max_2=[],[]
	min_1,min_2=[],[]

	total1=[]
	total2=[]
	for i in range(num_run):

		print i

		y1,y2 = estimator.gd_run(max_iter=max_iter,method=method)
		y1= concat(y1,max_iter)
		y2= concat(y2,max_iter)


		total1.append(y1)
		total2.append(y1)


	mean_1 = np.mean(total1,axis=0)
	mean_2 = np.mean(total2,axis=0)


	min_1 = np.amin(total1,axis=0)
	min_2 = np.amin(total2,axis=0)

	max_1 = np.amax(total1,axis=0)
	max_2 = np.amax(total2,axis=0)

	x=range(len(mean_1))
	y=range(len(mean_2))

	plt.figure()
	plt.subplot(2, 1, 1)
	plt.plot(x, mean_1, 'k-')
	plt.fill_between(x,min_1,max_1)
	plt.title('Evolution of likelihood and distance with iteration')
	plt.ylabel('Distance from True params')

	plt.subplot(2, 1, 2)
	plt.plot(y, mean_2, 'k-')
	plt.fill_between(x,min_2,max_2)
	plt.xlabel('iteration')
	plt.ylabel('Negative Log-Likelihood')

	plt.savefig("Plots/data_dim= "+str(data_dimension)+" sparsity= "+str(sparsity)+" T= "+str(T)+".png")
	plt.close()
	#plt.show()
		


for data_dimension in data_dimension_ar:
	for  sparsity in sparsity_ar:
		for T in T_ar:
			test_on_synthetic(1,data,data_dimension,sparsity,T)
			



















