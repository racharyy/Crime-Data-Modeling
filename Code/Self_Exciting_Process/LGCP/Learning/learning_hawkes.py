import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import bernoulli
from scipy.sparse import random
from parameters import *


class ClassName(langevin):
	"""docstring for ClassName"""
	def __init__(self,stepsize,N,count,prior_sig=2):
		super(ClassName, self).__init__()
		self.stepsize =stepsize
		self.method =method
		self.N =len(count)
		self.count=count



	def learning(self,method='MALA',max_iter=1000):

		#Initialize Parameters
		params = params(self.N,self.prior_sig)
		params.add_latent_param()
		params.add_intensity(self.count)

		if self.method == 'MALA':

			for i in range(max_iter):

				#Langevin Algorithm
				temp_dic={}
				der_dic=calc_der(params_dic['der_x'],lamb,params_dic['der_mu'],params_dic['der_sig'],params_dic['der_a'],self.N,self.count)

				#Metropolis Adjustment
				acc_prob=
				u = np.random.random()
				if u<acc_prob:
					params_dic = temp_dic










