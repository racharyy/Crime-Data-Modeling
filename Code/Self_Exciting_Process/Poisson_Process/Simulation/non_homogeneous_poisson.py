import numpy as np 
import matplotlib.pyplot as plt 
from homogeneous_poisson import *

class Non_homogeneous_poisson(object):
	"""docstring for Non_homogeneous_poisson"""
	def __init__(self, intensity,dimension,given_range):
		super(Non_homogeneous_poisson, self).__init__()
		self.intensity = intensity
		self.dimension = dimension
		self.range = given_range
	
	def intensity_function(self,):
		pass
	

	def max_intensity(self):

		pass


	def simulate(self,show_plot=True):
		

		hom_poisson_process = Homogeneous_poisson(intensity,self.dimension,self.given_range)
		hom_poisson_process.simulate()

		temp_data = hom_poisson_process.data 
		data = []
		for point in temp_data:
			u =np.random.random()
			accept_prob = float()/float(intensity)
			if u <= accept_prob:
				data.append(point)


		self.data=data
		if show_plot:
			if self.dimension ==1:

			
