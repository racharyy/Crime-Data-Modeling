import numpy as np 
import matplotlib.pyplot as plt 
from parameters import *
from math import ceil,isnan,log
from copy import copy
from scipy.optimize import fsolve
from auxilary import *



class Estimation_hawkes(object):
	"""docstring for Estimation_hawkes"""
	def __init__(self,data,background_num,true_param=None, issynthetic=True, method='EM',initialization='random'):
		super(Estimation_hawkes, self).__init__()
		self.method = method
		self.data = data
		self.issynthetic =issynthetic
		if issynthetic:
			self.true_param =true_param
		
		self.background_num = background_num
		self.data_length = data.shape[0]
		self.time_length = max(data[:,0])-min(data[:,0])
		self.given_range =[[min(data[:,1]),max(data[:,1])],[min(data[:,2]),max(data[:,2])]]

		#Partition the data in cluster_num many clusters using function from data_process
		self.cluster_event = [[[] for i in range(background_num)] for j in range(background_num)] 
		self.data_to_background = [[] for x in xrange(self.data_length)]
		xframe_len = ((self.given_range[0][1]-self.given_range[0][0])/self.background_num)
		yframe_len = ((self.given_range[1][1]-self.given_range[1][0])/self.background_num)
		print xframe_len,yframe_len
		self.cell_area = xframe_len*yframe_len
		dist_max = ((self.given_range[0][1]-self.given_range[0][0])**2+(self.given_range[1][1]-self.given_range[1][0])**2)
		for ind,point in enumerate(self.data):
			m = min(self.background_num-1,int((point[1]-self.given_range[0][0])/xframe_len))
			n = min(self.background_num-1,int((point[2]-self.given_range[1][0])/yframe_len))
			#print m,n,(point[1]-self.given_range[0][0]),(self.given_range[0][1]-self.given_range[0][0])
			self.cluster_event[m][n].append(ind) 
			self.data_to_background[ind]=(m,n)
		self.data_to_background = np.array(self.data_to_background)	
		print "Data partitioning done"

		#creating the time difference matrix and distance difference matrix which is necessary for EM algorithm
		dist_mat = np.zeros((self.data_length,self.data_length))
		time_diff_mat = np.zeros((self.data_length,self.data_length))

		for i in range(self.data_length):
			for j in range(i):
				dist_mat[i][j] = (1.0/ dist_max)*(((self.data[i][1]-self.data[j][1])**2 + (self.data[i][2]-self.data[j][2])**2))
				time_diff_mat[i][j] = (1.0/self.time_length)*(self.data[i][0]-self.data[j][0])
				


		self.dist_mat = np.array(dist_mat)
		self.time_diff_mat = np.array(time_diff_mat)
		print "time difference and dist difference matrix set up"

		
		#Initialize the parameters of the model
		if initialization=='random':
			K=np.random.random()
			time_offset=np.random.random()
			time_power=np.random.random()
			dist_offset=np.random.random()
			dist_power=np.random.random()

		else:
			K=1
			time_offset=0.1
			time_power=1
			dist_offset=0.1
			dist_power=1

		self.cur_param = params(K,time_offset,time_power,dist_offset,dist_power,np.random.random((self.background_num,self.background_num)),self.given_range)
		
		print "Initializing parametrs done"


		#Initializing the probability of being an event to be background or offspring
		vec_kernel = np.vectorize(self.cur_param.kernel)
		probability_matrix = np.tril(vec_kernel(self.time_diff_mat,self.dist_mat),-1)
		normal_row =1.0/( np.sum(probability_matrix,axis=1) + np.array([self.create_data_to_background(pair) for pair in self.data_to_background]))
		self.probability_matrix = np.dot(np.diag(normal_row),probability_matrix)
		self.cum_prob_back = np.sum(self.probability_matrix,axis=1)
		self.cum_prob_off = np.sum(self.probability_matrix,axis=0)
		print "Initializing probability matrix done"

	def create_data_to_background(self,pair):
		return self.cur_param.background[pair[0]][pair[1]]
	

	def update_offsets_power(self,expected_offspring):

		
		
		temp_time_offset, temp_dist_offset = 0.1,0.1
		time_offset,dist_offset = 0.1,0.1
		tol =0.01
		dist_flag =1
		time_flag =1

		time_mat = self.time_diff_mat+time_offset
		time_mat = np.tril(time_mat,-1)
		
					
		time_rhs1 =np.sum(vec_func(self.probability_matrix,time_mat))/float(expected_offspring)###### sustract first one
		time_rhs2 = np.sum(self.probability_matrix*vec_log(time_mat))/float(expected_offspring)######


		dist_mat = self.dist_mat+dist_offset
		dist_mat = np.tril(dist_mat,-1)
		
		
		dist_rhs1 =np.sum(vec_func(self.probability_matrix,dist_mat))/float(expected_offspring)###### sustract first one
		dist_rhs2 = np.sum(self.probability_matrix*vec_log(dist_mat))/float(expected_offspring)######

		def time_equation(time_offset):
			

			return f1(time_rhs1,time_rhs2,time_offset)
			
		def dist_equation(dist_offset):

			return f1(dist_rhs1,dist_rhs2,dist_offset)




		for iter in xrange(100):


			if time_flag:
				time_mat = self.time_diff_mat+time_offset
				time_mat = np.tril(time_mat,-1)

				time_rhs1 =np.sum(vec_func(self.probability_matrix,time_mat))/float(expected_offspring)
				time_rhs2 = np.sum(self.probability_matrix*vec_log(time_mat))/float(expected_offspring)

				time_offset = f1(time_rhs1,time_rhs2,time_offset)
				if time_offset<=0:
					time_offset = np.random.random()
				
				

			if dist_flag:
				dist_mat = self.dist_mat+dist_offset
				dist_mat = np.tril(dist_mat,-1)
				#print dist_mat
				
				dist_rhs1 =np.sum(vec_func(self.probability_matrix,dist_mat))/float(expected_offspring)
				dist_rhs2 = np.sum(self.probability_matrix*vec_log(dist_mat))/float(expected_offspring)

				dist_offset = f1(dist_rhs1,dist_rhs2,dist_offset)
				if dist_offset<=0:
					dist_offset = np.random.random()

			if abs(time_offset- temp_time_offset)<=tol:
				time_flag =0

			if abs(dist_offset- temp_dist_offset)<=tol:
				dist_flag =0
			
			if time_flag ==0 and dist_flag ==0:
				break

			temp_time_offset = copy(time_offset)
			temp_dist_offset = copy(dist_offset)

			#print dist_offset
		#Fixing params try----------------------------
		#time_offset = self.true_param.time_offset
		#Fixing params try----------------------------


		time_mat = self.time_diff_mat+time_offset
		time_mat = np.tril(time_mat,-1)

		dist_mat = self.dist_mat+dist_offset
		dist_mat = np.tril(dist_mat,-1)

		time_rhs1 =np.sum(vec_func(self.probability_matrix,time_mat))/float(expected_offspring)###### sustract first one		
		dist_rhs1 =np.sum(vec_func(self.probability_matrix,dist_mat))/float(expected_offspring)###### sustract first one
		

		time_power = -1.0/float(1-(1.0/float(time_rhs1*time_offset)))
		dist_power = -1.0/float(1-(1.0/float(dist_rhs1*dist_offset)))



		return time_offset,dist_offset,time_power,dist_power
		#return time_offset,dist_offset,time_power,dist_power



	


	def estimate_params(self,maxiter=100,tolerance=0.5):


		vec_kernel = np.vectorize(self.cur_param.kernel)

		if self.method == 'EM':
			error_ar =[]
			improv_ar =[]

			for iteration in range(maxiter):

				#if iteration%10 ==0:
				print "%d th iteration is going on ------" %iteration


				temp_param = copy(self.cur_param)
				#E step of the EM


				


				#Updating the probability of being an event to be background or offspring
				vec_kernel = np.vectorize(self.cur_param.kernel)
				probability_matrix = np.tril(vec_kernel(self.time_diff_mat,self.dist_mat),-1)
				normal_row =1.0/( np.sum(probability_matrix,axis=1) + np.array([self.create_data_to_background(pair) for pair in self.data_to_background]))
				self.probability_matrix = np.dot(np.diag(normal_row),probability_matrix)
				self.cum_prob_back =np.sum(self.probability_matrix,axis=1)
				self.cum_prob_off = np.sum(self.probability_matrix,axis=0)
				expected_offspring = np.sum(self.cum_prob_off)


				#Computing the the number of background events in each cell
				background_eventnum = np.zeros((self.background_num,self.background_num))
				for  cell_x in range(self.background_num):
					for cell_y in range(self.background_num):

						def op_back_prob(event_ind):
							return 1-self.cum_prob_back[event_ind]
						vec_op_back = np.vectorize(op_back_prob,otypes=[float])
						
						event_incellk = np.array(self.cluster_event[cell_x][cell_y])
						nk = np.sum(vec_op_back(event_incellk))
						background_eventnum[cell_x][cell_y]=nk

				#Computing the the number of offspring events 
				offspring_eventnum = self.cum_prob_off
				
					

				#M step of the EM


				#Update the background probability
				background_prob = background_eventnum*self.cell_area
				self.cur_param.background = background_prob

				#Updating time_power,dist_power and time_offset,dist_offset
				time_offset,dist_offset,time_power,dist_power = self.update_offsets_power(expected_offspring)
				


				#Updating K_0
				K = ((float((time_offset**time_power)*(dist_offset**dist_power)*time_power*dist_power))* float(expected_offspring))/(np.pi)#float(self.data_length*np.pi) #
				print K
				#K=1
				# time_offset = self.true_param.time_offset
				# time_power = self.true_param.time_power

				#Updating Parametrs
				self.cur_param.K =K
				self.cur_param.time_power =time_power
				self.cur_param.time_offset =time_offset
				self.cur_param.dist_power =dist_power
				self.cur_param.dist_offset =dist_offset

				#computing the error
				#print np.linalg.norm(temp_param.background- self.cur_param.background)
				improv = np.sqrt((temp_param.K-self.cur_param.K)**2 + (temp_param.time_power- self.cur_param.time_power)**2 + (temp_param.time_offset - self.cur_param.time_offset)**2 +(temp_param.dist_power - self.cur_param.dist_power)**2+ (temp_param.dist_offset - self.cur_param.dist_offset)**2 + np.linalg.norm(temp_param.background- self.cur_param.background))
				if self.issynthetic:
					error = np.sqrt((self.true_param.K-self.cur_param.K)**2 + (self.true_param.time_power- self.cur_param.time_power)**2 + (self.true_param.time_offset - self.cur_param.time_offset)**2 +(self.true_param.dist_power - self.cur_param.dist_power)**2+ (self.true_param.dist_offset - self.cur_param.dist_offset)**2 + np.linalg.norm(self.true_param.background- self.cur_param.background))
					error_ar.append(error)
					if error < tolerance and iteration >30:
						break
				#print error
				improv_ar.append(improv)
				
		#print error_ar
		if self.issynthetic:
			plt.subplot(1,2,1)
			plt.plot(range(len(error_ar)),error_ar)

			plt.subplot(1,2,2)
			plt.plot(range(len(improv_ar)),improv_ar)
		else:
			plt.plot(range(len(improv_ar)),improv_ar)

		#plt.show()
		return self.cur_param

		