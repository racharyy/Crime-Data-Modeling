import numpy as np 


class Triggering_events(object):
	"""docstring for Triggering_events"""
	def __init__(self, K,d,rho,c,omega):
		super(Triggering_events, self).__init__()
		self.K = K
		self.d = d
		self.rho =  rho
		self.c = c
		self.omega = omega



	def density_function(self,x,y,t):
		return float(self.K)/float(((x**2+y**2+self.d)**(1+self.rho))*((t+self.c)**(1+self.omega)))


	def normalization(self):
		return (float(self.K*np.pi)*((self.d**(-self.rho))*(self.c**(-self.omega))))/float(self.rho*self.omega)

	def sampling(self,root_time,root_x,root_y,num_samples):
		
		
		output = []
		current = (root_time,root_x,root_y)
		
		for iteration in xrange(10000+num_samples):
			pass:
			
			#Resample t
			u = np.random.random()
			t =  self.c*((1-u)**(-1.0/float(np.omega)) -1)
			current[0] =t

			#Resample x
			

			#Resample y



			#Add to output if mixes
			if iteration>10000:
				output.append(current)


		return output



# def mean_poisson(given_range,d,rho,num_points=10000):
# 	est=0
# 	for point in xrange(num_points):
# 		x= given_range[0][0] + (given_range[0][1]- given_range[0][1])*np.random.random()
# 		y= given_range[1][0] + (given_range[1][1]- given_range[1][1])*np.random.random()
# 		est = est + function(x,y,d,rho)
# 	return (1.0/float(10000))*est



