import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import bernoulli
from scipy.sparse import rand


class Autoregressive_model(object):
	"""docstring for Autoregressive_model"""
	def __init__(self, distribution = 'Poisson', nu =np.zeros(5)  ,range=[-1,0] , data_dimension=5,sparsity=10):
		super(Autoregressive_model, self).__init__()
		self.distribution = distribution
		self.nu = nu
		self.params = rand(data_dimension,data_dimension,density=float(sparsity)/float(data_dimension**2)).A
		self.data = []
		self.range = range
		self.data_dimension = data_dimension



	def generate(self,T):

		if self.distribution == 'Poisson':
			X1 = np.random.poisson(lam=np.ones(self.data_dimension),size=self.data_dimension)
			self.data.append(X1)
			if T>1:
				Xlast = X1
				for i in range(T):
					lam = np.exp(self.nu-np.dot(self.params,Xlast))
					Xi = np.random.poisson(lam)
					self.data.append(Xi)
					Xlast = Xi




		elif self.distribution == 'Bernouli':
			X1 = bernoulli(p=0.5,size=self.data_dimension)
			self.data.append(X1)
			if T>1:
				Xlast = X1
				for i in range(T-1):
					p = 1.0/float(1+np.exp(nu-np.dot(self.params,Xlast)))
					Xi = bernoulli(p,size=self.data_dimension)
					self.data.append(Xi)
					Xlast = Xi

		