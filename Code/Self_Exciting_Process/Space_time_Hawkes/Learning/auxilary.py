import numpy as np 
import matplotlib.pyplot as plt 
from parameters import *
from math import ceil,isnan,log
from copy import copy
from scipy.optimize import fsolve




def nonz_log(x):
	if x==0:
		return 0
	else:
		return log(x)
	

vec_log = np.vectorize(nonz_log,otypes=[float])

def basic_func(a,b):
	if b==0:
		return 0
	else:
		return a/float(b)

vec_func = np.vectorize(basic_func,otypes=[float])


def f1(coeff1,coeff2,inp):

	if coeff1==0:
		print 'fuck'

	
	return 1.0/float(coeff1*(coeff2+1-log(inp)))


# def update_offsets_power(self,expected_offspring):

		
	# 	time_offset = max(self.cur_param.time_offset,0.1)
	# 	dist_offset = max(self.cur_param.dist_offset,0.1)

	# 	vec_log = np.vectorize(log)

	# 	def basic_func(a,b):
	# 		return a/float(b)

	# 	vec_func = np.vectorize(basic_func)
		
	# 	for iters in range(10):
	# 		if time_offset<=0:
	# 				time_offset = np.random.random()

	# 		if dist_offset<=0:
	# 			dist_offset = np.random.random()

	# 		#print time_offset,dist_offset

	# 		time_rhs1 =np.sum(vec_func(self.probability_matrix,self.time_diff_mat+time_offset))/float(expected_offspring)###### sustract first one
	# 		time_rhs2 = np.sum(self.probability_matrix*vec_log(self.time_diff_mat+time_offset))/float(expected_offspring)######

	# 		dist_rhs1 =np.sum(vec_func(self.probability_matrix,self.dist_mat+dist_offset))/float(expected_offspring)###### sustract first one
	# 		dist_rhs2 = np.sum(self.probability_matrix*vec_log(self.dist_mat+dist_offset))/float(expected_offspring)######

			
	# 		def f1(coeff1,coeff2,inp):
	# 			if inp<=0:
	# 				print 'fuck'
	# 			return 1.0/float(coeff1*inp) + log(inp)-coeff2-1

	# 		def der_f1(coeff1,inp):
	# 			return -float(1)/(coeff1*(inp**2)) + 1.0/inp


	# 		for sub_iter in range(5):
	# 			if time_offset<=0:
	# 				time_offset = np.random.random()

	# 			if dist_offset<=0:
	# 				dist_offset = np.random.random()
	# 			time_temp = copy(time_offset)
	# 			dist_temp = copy(dist_offset)
	# 			time_offset = time_offset -float(f1(time_rhs1,time_rhs2,time_offset))/float(der_f1(time_rhs1,time_offset))
	# 			dist_offset = dist_offset -float(f1(dist_rhs1,dist_rhs2,dist_offset))/float(der_f1(dist_rhs1,dist_offset))
	# 			if abs(time_offset-time_temp)<=0.1 and abs(dist_offset-dist_temp)<=0.1:
	# 				break
	# 		if time_offset<=0:
	# 				time_offset = np.random.random()

	# 		if dist_offset<=0:
	# 			dist_offset = np.random.random()	

	# 		time_power = 1.0/float(1-1.0/float(time_rhs1*time_offset))
	# 		dist_power = 1.0/float(1-1.0/float(dist_rhs1*dist_offset))



	#	 return time_offset,dist_offset,time_power,dist_power