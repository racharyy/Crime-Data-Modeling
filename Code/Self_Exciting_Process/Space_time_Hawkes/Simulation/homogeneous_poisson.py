import numpy as np 
import matplotlib.pyplot as plt 

class Homogeneous_poisson(object):
	"""docstring for Homogeneous_poisson"""
	def __init__(self, intensity,dimension,given_range):
		super(Homogeneous_poisson, self).__init__()
		self.intensity = intensity
		self.dimension = dimension
		self.range = given_range


	def simulate(self,show_plot=True):
		
		if self.dimension ==1:
			data=[]
			t = self.range[0]
			while t <= self.range[1]:
				u = np.random.random()
				t=t-(np.log(u)/self.intensity)
				if t>self.range[1]:
					break
				data.append(t)

	
			self.data=data


			if show_plot:

				plt.subplot(2,1,1)
				plt.plot(data,np.zeros(len(data)),'g.')
				plt.xlabel('time of events')
				plt.title('Occuring of events')

				plt.subplot(2,1,2)
				n, bins, patches = plt.hist(data, 10, facecolor='g', alpha=0.75)				
				plt.xlabel('time')
				plt.ylabel('number of events')
				plt.title('Histogram of number of events')
				#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
				plt.grid(True)
				plt.show()



				
		
		elif self.dimension==2:

			x_int = self.intensity*abs(self.range[1][1]-self.range[1][0])
			x_data=[]
			x = self.range[0][0]
			while x<=self.range[0][1]:
				u = np.random.random()
				x=x-(np.log(u)/self.intensity)
				if x>self.range[0][1]:
					break
				x_data.append(x)
			n= len(x_data)
			y_data=[]
			for i in xrange(n):
				y_data.append(self.range[1][0]+(abs(self.range[1][1]-self.range[1][0]))*np.random.random())

			data = zip(x_data,y_data)
			self.data=data

			if show_plot:



				# fig, axs = plt.subplots(2, 1,  sharex=True, sharey=True,tight_layout=True)
				# axs[0].plot(x_data,y_data,'g.')
				# plt.xlabel('time of events')
				# plt.title('Occuring of events')


				bins=plt.hist2d(x_data,y_data, bins=100)
				plt.colorbar()
				#bins = axs[1].hist2d(x_data,y_data, bins=40)
				# plt.xlabel('x_axis')
				# plt.ylabel('y_axis')
				# plt.title('Histogram of number of events')
				plt.show()


# intensity = 10
# given_range =[[0,100],[0,100]]
# dimension=2

# hom_poi = Homogeneous_poisson(intensity,dimension,given_range)
# hom_poi.simulate()




