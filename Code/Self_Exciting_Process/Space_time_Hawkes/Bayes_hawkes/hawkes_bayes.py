import matplotlib.pyplot as pyplot
import numpy as np 
from scipy.stats import norm 
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from pyhmc import hmc
import math
import theano as tt

class Bayes_Hawkes(object):
	"""docstring for Bayes_Hawkes"""
	def __init__(self, data):
		super(Bayes_Hawkes, self).__init__()
		self.data = data
		self.data_len = data.shape[0]
		self.time_length = max(data[:,0])-min(data[:,0])
		self.given_range =[[min(data[:,1]),max(data[:,1])],[min(data[:,2]),max(data[:,2])]]

		x_range = self.given_range[0][1]-self.given_range[0][0]
		y_range = self.given_range[1][1]-self.given_range[1][0]

		min_time = min(data[:,0])
		print 'data loaded'

		#Transform the data to center w.r.t origin
		self.transformed_data = np.array([[0,0,0] for i in range(self.data_len)])
		for x in xrange(self.data_len):
			self.transformed_data[x][0] = data[x][0]-min_time
			self.transformed_data[x][1] = data[x][1]-(self.given_range[0][0]-x_range/2.0)
			self.transformed_data[x][2] = data[x][2]-(self.given_range[1][0]-y_range/2.0)
		self.transformed_data = self.transformed_data.astype(np.float64)
		
		self.transformed_range = [[-x_range/2.0,x_range/2.0],[-y_range/2.0,y_range/2.0]]
		self.x_range = self.transformed_range[0][1]-self.transformed_range[0][0]
		self.y_range = self.transformed_range[1][1]-self.transformed_range[1][0]
		
		print 'data transformation done'


		#Time difference and space difference matrix needed fot the intensity function
		time_mat = np.zeros((self.data_len,self.data_len))
		space_mat = np.zeros((self.data_len,self.data_len))
		for i in xrange(self.data_len):
			for j in xrange(i):
				time_mat[i][j] = self.transformed_data[i][0]-self.transformed_data[j][0]
				space_mat[i][j] = (self.transformed_data[i][1]-self.transformed_data[j][1])**2 + (self.transformed_data[i][2]-self.transformed_data[j][2])**2
		self.time_mat = time_mat
		self.space_mat = space_mat
		print 'time_mat,space_mat creation done'
		
		#there are 4 parameters m_0,theta,sigma,omega. Hence we have gradiant of size 4
		self.grad=np.array([0,0,0,0])

	#Fitting the background to get the best bandwidth for the 'epanechnikov' kernel
	def fit_background(self):

		min_xy = min(self.x_range/2,self.y_range/2)
		sig_xy=np.log10(np.sqrt(2)*min_xy)

		sig_t = np.log10(self.time_length)

		data_x = self.transformed_data[:,1]
		data_y = self.transformed_data[:,2]

		time_data = (self.transformed_data[:,0]).reshape((self.data_len,1))
		space_data = np.array([data_x,data_y]).T
		#print space_data.shape,time_data.shape
		band_space = {'bandwidth': np.logspace(sig_xy/10, sig_xy, 20)}
		grid_space = GridSearchCV(KernelDensity(kernel = 'epanechnikov'), band_space)
		grid_space.fit(space_data)
		space_kde = grid_space.best_estimator_#['bandwidth']
		print 'spatial kernel smoothing done'

		band_time = {'bandwidth': np.logspace(sig_t/10, sig_t, 20)}
		grid_time = GridSearchCV(KernelDensity(kernel = 'epanechnikov'), band_time)
		grid_time.fit(time_data)
		time_kde = grid_time.best_estimator_#['bandwidth']
		print 'temporal kernel smoothing done'

		self.back_space_bw = space_kde.bandwidth.astype(np.float64)
		self.back_time_bw = time_kde.bandwidth.astype(np.float64)

		
		#temporal part in the background intensity
		max_t = min(self.time_length,self.back_time_bw)
		integral_time_part = 0.75*max_t-(0.25*max_t**3)/self.back_time_bw**2

		#spatial part in the background intensity
		min_rad = min(self.back_space_bw,min_xy)
		integral_space_part = 2*np.pi*min_rad - 0.25*(min_rad**3)/(self.back_space_bw**2)

		def ker(stamp):
			t,x,y = stamp
			time_part = (0.75*(1-(t/self.back_time_bw)**2)*(t<=self.back_time_bw))
			space_part = (0.75*(1-(x/self.back_space_bw)**2-(y/self.back_space_bw)**2)*((x**2+y**2)<=self.back_space_bw**2))
			return time_part * space_part
		
		back_data_int = np.array([ker(stamp) for stamp in self.transformed_data])
		
		#print back_data_int,integral_time_part,integral_space_part,'check background'

		
		self.back_intensity_data =  np.maximum(0.00000001,back_data_int).astype(theano.config.floatX)
		self.back_intensity_integral = (integral_time_part*integral_space_part).astype(theano.config.floatX)
		#print (self.back_intensity_data)
		#print np.sum(np.log(self.back_intensity_data)),self.back_intensity_integral
		
		#self.grad[0] = self.likelihood_background_part



	#contribution of the offspring kernel in the log likelihood
	def likelihood_offspring(self,params):

		data_x = self.transformed_data[:,1]
		data_y = self.transformed_data[:,2]

		time_data = self.transformed_data[:,0]

		m,theta,sigma,omega = params

		#offspring data part in the likelihood
		time_ker = np.tril(np.exp(-omega*self.time_mat),-1)
		space_ker = np.tril((1.0/(2*np.pi*sigma**2))*np.exp((-0.5/sigma**2)*self.space_mat),-1)

		#####TODO#########
		data_ker = (np.sum(time_ker*space_ker,axis=1)).astype(theano.config.floatX)

		#offsping intensity part of the likelihood
		max_t= max(time_data)
		time_diff = max_t - np.array(time_data)
		time_integral =(1.0/(omega))*(1-np.exp(-omega*(time_diff)))
		# print np.sum(time_ker),np.sum(time_integral),'check time'

		min_x= self.transformed_range[0][0]
		max_x= self.transformed_range[0][1]
		min_y= self.transformed_range[1][0]
		max_y= self.transformed_range[1][1]

		def norm_range_x(x):
			return norm.cdf(max_x,loc=x,scale=sigma)-norm.cdf(min_x,loc=x,scale=sigma)
		vec_norm_range_x = np.vectorize(norm_range_x,otypes=[np.float64])

		def norm_range_y(x):
			return norm.cdf(max_y,loc=x,scale=sigma)-norm.cdf(min_y,loc=x,scale=sigma)
		vec_norm_range_y = np.vectorize(norm_range_y,otypes=[np.float64])

		space_integral = vec_norm_range_x(data_x)*vec_norm_range_y(data_y)*range(self.data_len)[::-1]
		# print np.sum(space_ker),np.sum(space_integral),'check space'

		integral_ker = (np.sum(time_integral *space_integral)).astype(theano.config.floatX)
		# print 'data_ker',data_ker,integral_ker

		# #required calculation for the derivative wrt sigma
		# space_der = -np.tril((2.0/(sigma))*(np.ones((self.data_len,self.data_len))-(1.0/sigma**2)*self.space_mat),-1)
		# data_der = np.sum(space_der*space_ker*time_ker)
		

		# #derivative wrt sigma for the integral
		# space_der_integral_x = np.array([-((max_x-xs)/(sigma**2))*norm.pdf(max_x,loc=xs,scale=sigma)+((min_x-xs)/(sigma**2))*norm.pdf(min_x,loc=xs,scale=sigma) for xs in data_x])
		# space_der_integral_y = np.array([-((max_y-ys)/(sigma**2))*norm.pdf(max_y,loc=ys,scale=sigma)+((min_y-ys)/(sigma**2))*norm.pdf(min_y,loc=ys,scale=sigma) for ys in data_y])
		# der_integral = np.sum(space_der_integral_x*space_der_integral_y*time_integral)

		#Computing opossible gradients
		# self.grad[1] = omega*(data_ker-integral_ker)
		# self.grad[2] = theta*(data_ker*(1-omega)+integral_ker*(omega**2))
		# self.grad[3] = theta*omega*(data_der-der_integral) 


		self.intensity_data = np.sum(np.log(m*self.back_intensity_data + theta*omega*data_ker))
		self.intensity_integral = m*self.back_intensity_integral + theta*omega*integral_ker

		




	#computing likelihood using background and offspring intensity
	def log_likelihood(self,params):

		self.likelihood_offspring(params)
		print 'data_intensity',self.intensity_data, 'integral_intensity', self.intensity_integral
		return self.intensity_data - self.intensity_integral

	#Normal prior for the parameters m,theta,sigma,omega
	def log_prior(self,params):

		m,theta,sigma,omega = params
		return (2*norm.logpdf(theta,scale=np.sqrt(10))+ 2*norm.logpdf(sigma,scale=np.sqrt(10))+2*norm.logpdf(m,scale=1)+2*norm.logpdf(omega,scale=np.sqrt(10))).astype(theano.config.floatX)


	#posterior = prior*likelihood
	def log_posterior(self,params):

		
		
		return self.log_prior(params)- self.log_likelihood(params)  #, self.grad 

	#HMC sampler to sample parametrs of the model
	def posterior_sampling(self,n_samples=10):


		#init_params = np.array([abs(np.random.normal()),abs(np.random.normal(scale=np.sqrt(10))),abs(np.random.normal(scale=np.sqrt(10))),abs(np.random.normal(scale=np.sqrt(10)))])
		print self.log_posterior (init_params)
		






