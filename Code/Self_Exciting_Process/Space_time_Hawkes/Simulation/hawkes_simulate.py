from __future__ import absolute_import
import numpy as np 
import matplotlib.pyplot as plt
#from ..learning import basic_hawkes
#from Self_Exciting_Process.Poisson_Process.Simulation import homogeneous_poisson
from ...poisson_Process.simulation.homogeneous_poisson import *
from math import ceil
from ..learning.parameters import *


class Spatio_temp_hawkes(object):
	"""docstring for patio_temp_hawkes"""
	def __init__(self, given_range,cell_division,max_intensity,T,params_type='random'):
		super(Spatio_temp_hawkes, self).__init__()
		self.range = given_range
		self.xdivision = np.linspace(given_range[0][0],given_range[0][1],cell_division)
		self.ydivision = np.linspace(given_range[1][0],given_range[1][1],cell_division)
		self.data=[]
		self.max_T =T
		background = max_intensity*np.random.random((cell_division-1,cell_division-1)) 
		K=1
		self.cell_division = cell_division
		# dist_max = np.sqrt((given_range[0][1]-given_range[0][0])**2+(given_range[1][1]-given_range[1][0])**2)
		# print "hello",dist_max
		if params_type == 'random':

			time_offset = np.random.random()
			time_power = np.random.random()
			dist_power = np.random.random()
			dist_offset = np.random.random()

		else:

			time_offset =0.5
			time_power =0.5
			dist_power =0.5
			dist_offset =0.5

		self.params = params(K,time_offset,time_power,dist_offset,dist_power,background,given_range)
		
		

	def simulate(self):

		data_background=[]
		#Initial step of generating background events
		for x in xrange(1,self.cell_division):
			for y in xrange(1,self.cell_division):
				#print x,y
				sub_range = [[self.xdivision[x-1],self.xdivision[x]],[self.ydivision[y-1],self.ydivision[y]]]
				#print self.params.background[x-1][y-1],sub_range
				back_pois = Homogeneous_poisson(self.params.background[x-1][y-1],2,sub_range)
				back_pois.simulate(show_plot=False)
				data_background = data_background+back_pois.data		
		for data in data_background:
			time = self.max_T*np.random.random()
			self.data.append((time,data[0],data[1]))


		#generating the offspring events
		for events in self.data:
			
			num_events = np.random.poisson(self.params.offspring_mean())
			point_diff = self.params.sample(num_events,self.range,self.max_T)
			for points in point_diff:
				time =events[0]+ points[0]
				ux=np.random.random()
				uy=np.random.random()
				if ux<0.5:
					cur_x = events[1]+points[1]
				else:
					cur_x = events[1]-points[1]


				if uy<0.5:
					cur_y = events[2]+points[2]
				else:
					cur_y = events[2]-points[2]

				if (time<=self.max_T and time>=0) and (cur_x<= self.range[0][1] and cur_x>= self.range[0][0]) and (cur_y<= self.range[1][1] and cur_y>= self.range[1][0]):

					self.data.append((time,cur_x,cur_y))

		self.data.sort()
		#print len(self.data)




				
		