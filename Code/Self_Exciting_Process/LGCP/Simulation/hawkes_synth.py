import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import bernoulli
from scipy.sparse import random
from parameters import *

class Hawkes(object):
	"""docstring for Hawkes"""
	def __init__(self, N,prior_sig):
		super(Hawkes, self).__init__()
		params = params(N,prior_sig)
		params.add_latent_param()
		self.params =params


	def generate_data(self):
		data_set = np.zeros(self.params.N) 
		self.lamb[0] = np.exp(params.mu)
		data_set[0] = np.random.poisson(lamb, 1)
		const = self.theta*(float(1-self.b)/float(self.b))
		for i in range(1,params.N):
			lamb = lamb+np.exp(self.latent_x[i])+const*(self.b)*data_set[i-1]-np.exp(self.latent_x[i-1])
			self.lamb_ar[i] = lamb
			data_set[i] = np.random.poisson(lamb, 1)

		self.data_set =data_set



	def plot_data(self):
		xaxis = range(self.params.N)
		yaxis = self.data_set
		zaxis = self.lamb_ar

		plt.figure()
		plt.subplot(2, 1, 1)
		plt.plot(xaxis, yaxis, 'k-')
		plt.title('Evolution of Intensity and Number of Events with iteration')
		plt.ylabel('Number of Events')

		plt.subplot(2, 1, 2)
		plt.plot(xaxis, zaxis, 'k-')
		plt.xlabel('iteration')
		plt.ylabel('Intensity')

		plt.show()

		



