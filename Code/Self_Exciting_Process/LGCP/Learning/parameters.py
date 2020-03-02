import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import bernoulli
from scipy.sparse import rand
from scipy.special import logsumexp

#Auxilary Function for calculating the derivative for the update
class params(object):
	"""docstring for params"""
	def __init__(self,N,prior_sig):
		super(params, self).__init__()

		self.mu = prior_sig*np.random.randn(1)
		self.sigma =prior_sig*np.random.randn(1)
		self.a = np.random.random(1)
		self.b = np.random.random(1)
		self.theta = np.random.random(1)
		self.N = N
		self.x=[]
		self.lamb=np.zeros(N)

		
	#Create the covariance matrix with $\Sigma_{ij}=\sigma^2*a^{|i-j|}$
	def create_cov_mat():
		mat = np.zeros((N,N))
		for i in range(N):
			for j in range(i+1):
				x=sigma*(a**(abs(i-j)))
				mat[i][j]=x

		return mat

	#Add the latent x parameters
	def add_latent_param():
		cov_mat = self.create_cov_mat()
		self.x=np.random.multivariate_normal(self.mu*np.ones(self.N), cov_mat,1)[0]

	#Add the intensity values
	def add_intensity(count):
		lamb_temp=np.exp(params.mu)
		self.lamb[0] = lamb_temp
		const = self.theta*(float(1-self.b)/float(self.b))
		for i in range(1,self.N):
			lamb_temp = lamb_temp+np.exp(self.x[i])+const*(self.b)*count[i-1]-np.exp(self.x[i-1])
			self.lamb[i] = lamb_temp


	#making of the inverse matrix
	def create_inv_matrix():
		mat = np.zeros((self.N,self.N))
		for i in range(self.N):
			if i==0 or i==self.N-1:
				mat[i][i]=1
				mat[i][i+1]=-self.a
			else:
				mat[i][i]=1+self.a**2
				mat[i][i-1]=-self.a
				mat[i-1][i]=-self.a

			mat[self.N-1][self.N-2]=-self.a
		mat=float(1)/float(self.sigma*(1-self.a**2))
		return mat

	#Calculating determinant of the covariance matrix of the gaussian process	
	def det_sigmat():
		return (self.sigma**self.N)*((1-self.a**2)**(self.N-1))

	#derivative of the covariance matrix wrt a
	def der_mat_a():
		return np.fromfunction(lambda i, j: abs(i-j)*sigma*(a**(abs(i-j)-1)), (self.N,self.N))
	
	#derivative of  the covariance matrix wrt sigma			
	def der_mat_sig():
		return np.fromfunction(lambda i, j: a**(abs(i-j)), (self.N,self.N))

	
	#derivative of lambda wrt b
	def der_lamb_b():
		res=np.zeros(N+1)
		coeff_ar= np.array([0]+[(float(self.count[i])/float(self.lamb[i]) -1) for i in range(self.N)])
		for i in range(1,self.N+1):
			res[i] = (self.b*res[i-1]+self.lamb[i-1]-self.theta*self.count[i-1])
		return sum(res*coeff_ar)

	#derivative of lambda wrt theta
	def der_lamb_theta():
		res=np.zeros(N+1)
		coeff_ar= np.array([0]+[(float(count[i])/float(lamb[i]) -1) for i in range(N)])
		for i in range(1,N+1):
			res[i] = (b*res[i-1]+(1-b)*count[i-1])
		return sum(res*coeff_ar)

	def calc_der():

		#creating necessary matrices and vectors
		inv_sig = create_inv_matrix(sigma,a,N)
		det_sig = det_sigmat(sigma,a,N)
		mu_ar = mu*np.ones(N)
		mat_der_a =der_mat_a(a,sigma,N)
		mat_der_sig = der_mat_sig(a,sigma,N)
		part = np.dot(inv_sig,(x-mu_ar))

		#A dictionary with required derivatives
		der_dic={}
		
		#Calculating the derivatives
		der_x = np.exp(x)*(y*(1.0/lamb)-np.ones(N)) - part
		der_a = (0.5*det_sig*(N-1)*2a)/(1-a**2) + 0.5*np.dot(part.T,np.dot(mat_der_a,part))+
		der_sig = (-0.5*det_sig*N)/(sigma)+0.5*np.dot(part.T,np.dot(mat_der_sig,part))+
		der_mu = np.sum(part)+
		der_b = der_lamb_b(lamb,count,b,N)+
		der_theta = der_lamb_theta(lamb,count,b,N)+

		#Assigning values to the dictionary
		der_dic['der_x']=der_x
		der_dic['der_a']=der_a
		der_dic['der_sig']=der_sig
		der_dic['der_mu']=der_mu
		der_dic['der_b']=der_b
		der_dic['der_theta']=der_theta

		return der_dic


	def calc_post(x,lamb,mu,sigma,a,N,count):



		log_ans=logsumexp(-lamb)+logsumexp(count*np.log(lamb))-0.5*N*np.log(sigma)-0.5*(N-1)*np.log(1-a**2)

		























