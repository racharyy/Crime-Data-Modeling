import numpy as np 
import matplotlib.pyplot as plt 




class params(object):
	"""docstring for params"""
	def __init__(self, K,time_offset,time_power,dist_offset,dist_power,background,given_range):
		super(params, self).__init__()
		self.K = K
		self.time_offset =time_offset
		self.time_power =time_power
		self.dist_power =dist_power
		self.dist_offset =dist_offset
		self.background = background
		self.given_range = given_range 
		
	
	def kernel(self,time_diff,dist):
		
		denom = ((time_diff+self.time_offset)**(1+self.time_power))*((dist+self.dist_offset)**(1+self.dist_power))
		#print denom,self.time_offset,time_diff ,self.dist_offset,dist
		return float(self.K)/(float(denom))


	
	def offspring_mean(self):
		
		return self.K*np.pi*(((self.time_offset*(-self.time_power))*(self.dist_offset*(-self.dist_power)))/(self.dist_power*self.time_power))

				
	def sample(self,num_samples,given_range,T):

		#initialization randomly
		cur_t = T*np.random.random()
		cur_x = given_range[0][0]+ (given_range[0][1]-given_range[0][0])*np.random.random()
		cur_y = given_range[1][0]+ (given_range[1][1]-given_range[1][0])*np.random.random()
		cur_dist = cur_x**2+cur_y**2
		samples = np.zeros((num_samples,3)) 

		for iterations in range(num_samples):

			#Proposed sample
			t_star,x_star,y_star =  np.array([cur_t,cur_x,cur_y])+np.random.normal(size=3)
			dist_star = x_star**2+y_star**2
			
			#Metroplois Filter
			if np.random.random()< self.kernel(t_star,dist_star)/self.kernel(cur_t,cur_dist):
				cur_t,cur_x,cur_y = t_star,x_star,y_star
				cur_dist = dist_star
			samples[iterations] = (cur_t,cur_x,cur_y)

		return samples



	#Returns a matrix where A[i,j] = Pr[u_i=j], i.e., prob that ith event is triggered by jth event
	def required_matrix(self,data):

		xdim= self.background.shape[0]
		ydim= self.background.shape[1]
		xframe_len = (self.given_range[0][1]-self.given_range[0][0])/xdim
		yframe_len = (self.given_range[1][1]-self.given_range[1][0])/ydim
		N= data.shape[0]
		data =np.sort(data)
		prob_mat = np.zeros((N,N))
		time_diff_mat = np.zeros((N,N))
		dist_mat = np.zeros((N,N))
		lambda_ar = np.zeros(N)
		for i in range(N):
			m = int(data[i][1]/xframe_len)
			n = int(data[i][2]/yframe_len)
			prob_mat[i][i] = self.background[m][n]
			for j in range(i):
				time_diff = data[i][0]-data[j][0]
				dist = (data[i][1]-data[j][1])**2+(data[i][2]-data[j][2])**2
				prob_mat[i][j]=self.kernel(time_diff,dist)
				time_diff_mat[i][j]=time_diff+self.time_offset
				dist_mat[i][j]=dist+self.dist_offset

			lambda_ar[i]=np.sum(prob_mat[i,:])

		return np.dot(np.diag(lambda_ar),prob_mat),time_diff_mat,dist_mat

	

		
