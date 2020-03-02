import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import bernoulli
from scipy.sparse import rand
from copy import copy
from ar_synthetic_generate import *

class rmle(object):
	"""docstring for rmle"""
	def __init__(self,data,nu,data_dimension, sparsity,T,issynthetic=True,model='Poisson'):
		super(rmle, self).__init__()
		
		self.nu = nu
		self.issynthetic = issynthetic
		if self.issynthetic:
			synth = Autoregressive_model(distribution = model, nu =nu, data_dimension=data_dimension, sparsity=sparsity)
			synth.generate(T)
			data = synth.data
			self.data = np.array(data)
			data_source = synth
		else:
			data = np.array(data)
			assert(data.shape[1]==nu.shape[0])
			self.data = data
			data_source = "User"

		self.data_source = data_source
		self.model = model
		self.data_dimension = self.data.shape[1]
		self.T = self.data.shape[0]
		self.lamda = 0.1/(np.sqrt(self.T))


	def gradient(self,A):

		assert(A.shape[0]==A.shape[1] and A.shape[0]== self.data.shape[1])
		top_t = np.array(self.data[:-1])
		bottom_t = np.array(self.data[1:])
		

		if self.model=='Poisson':
			
			part1 = np.dot(np.transpose(bottom_t),top_t)
			mat =  np.exp(-np.dot(A,np.transpose(top_t)))
			part2 = np.dot(mat,top_t)
			part2 = np.dot(np.diag(np.exp(self.nu)),part2)

			return (1.0/float(self.T))* (part1-part2)

	def proximal(self,A,rate):

		assert(A.shape[0]==A.shape[1] and A.shape[0]== self.data.shape[1])
		return np.sign(A)*np.maximum(np.zeros(np.shape(A)),np.abs(A)-rate*np.ones(np.shape(A)))

	def likelihood(self,A):

		assert(A.shape[0]==A.shape[1] and A.shape[0]== self.data.shape[1])
		top_t = np.array(self.data[:-1])
		bottom_t = np.array(self.data[1:])

		if self.model =='Poisson':

			temp = np.dot(A,np.transpose(top_t))
			part1 = temp*(np.transpose(bottom_t))
			part2 = np.dot(np.diag(np.exp(self.nu)),np.exp(-temp))
			return (1.0/float(self.T))*np.sum(part1+part2)


	def minimization_function(self,A,Ax_old,rate):


		p1 = self.likelihood(Ax_old)
		p2 = np.sum(np.inner(self.gradient(Ax_old),A-Ax_old))
		p3 = (0.5/rate)* (np.linalg.norm(A-Ax_old)**2)

		
		
		return p1+p2+p3
		


	def gd_run(self,method='prox_gd',max_iter=100,threshold=0.2):

		#Intialize variables 
		#Ax_old = np.zeros((self.data_dimension,self.data_dimension))
		Ax_old = rand(self.data_dimension,self.data_dimension).A
		Ay_old = copy(Ax_old)

		dist = np.inf
		t_old =1
		neg_ll_ar= []
		dist_ar = []
		iteration =0
		beta = 0.75


		if method == 'acc_prox_gd':

			while (iteration < max_iter and threshold < dist):

				if iteration%100 ==0:
					print iteration
				rate = 0.01

				#Line Search
				while True:

					#Accelerated GD step
					temp = Ay_old - rate*self.gradient(Ay_old)
					Ax_new = self.proximal(temp,rate*self.lamda)
					a=self.minimization_function(Ax_new,Ax_old,rate)
					#a=self.minimization_function(Ax_new,Ay_old,rate)
					if self.likelihood(Ax_new) <= a:
						break
					else:
						rate = rate*beta
					

				#the proximal operation
				t_new = 0.5+0.5*np.sqrt(1+4*(t_old**2))
				Ay_new = Ax_new + float((t_old-1)/(t_new))*(Ax_new-Ax_old)
				

				#Check for convergence
				if self.issynthetic:
					dist = np.linalg.norm(Ay_new - self.data_source.params)
					dist_ar.append(dist)
				ll = self.likelihood(Ay_new)
				neg_ll_ar.append(ll)
				if dist <= threshold:
					break
				

				#Update values for next step
				Ay_old = copy(Ay_new)
				Ax_old = copy(Ax_new)
				t_old = copy(t_new)
				iteration =iteration+1
				

				#print dist



		if method == 'prox_gd':

			while (iteration < max_iter and threshold < dist):

				

				#Line Search

				rate = 0.01
				while True:

					#the proximal operation
					temp = Ax_old - rate*self.gradient(Ax_old)
					Ax_new = self.proximal(temp,rate*self.lamda)
					a=self.minimization_function(Ax_new,Ax_old,rate)


					if self.likelihood(Ax_new) <= a:
						#print 'success'
						break
					else:
						#print 'fail'
						rate = rate*beta
		

				#Check for convergence
				if self.issynthetic:
					dist = np.linalg.norm(Ax_new - self.data_source.params)
					dist_ar.append(dist)
				neg_ll_ar.append(self.likelihood(Ax_new))
				if dist <= threshold:
					break
				

				#Update values for next step
				Ax_old = copy(Ax_new)
				iteration =iteration+1
				

	


		


		return dist_ar,neg_ll_ar

		





				





