import numpy as np 
import matplotlib.pyplot as plt 
from generate_synth import *
import sys
sys.path
sys.path.append('./')
from Self_Exciting_Process.space_time_Hawkes.learning.hawkes_EM import *
import seaborn as sns


#Process True Data
data=np.loadtxt(open("../../Data/elmgrove_end_all.csv", "rb"), delimiter=",", skiprows=1)[:100]


#Fit True Data
background_num=10
hawkes_model = Estimation_hawkes(data,background_num,issynthetic=False)
estimated_params=hawkes_model.estimate_params()
each_block =10
total_pts =background_num*each_block

def func_intensity(time,num_xy =total_pts):
	given_range = hawkes_model.given_range
	x_points = np.linspace(given_range[0][0],given_range[0][1],num_xy)
	y_points = np.linspace(given_range[1][0],given_range[1][1],num_xy)
	Z = np.zeros((num_xy,num_xy))
	for ind_x,x in enumerate(x_points):
		for ind_y,y in enumerate(y_points):
			back_ind_x,back_ind_y = int(float(ind_x)/each_block), int(float(ind_y)/each_block)
			intensity = estimated_params.background[back_ind_x][back_ind_y]
			for points in hawkes_model.data:
				if points[0]<time:
					time_diff = time -points[0]
					dist = (x-points[1] )**2 + (y - points[2] )**2
					intensity = intensity+estimated_params.kernel(time_diff,dist)
			Z[ind_x][ind_y]= intensity
	return Z

time_length = hawkes_model.time_length
num_frams=5

for x in xrange(num_frams):
	time = (time_length/num_frams)*x +min(hawkes_model.data[:,0])
	data = func_intensity(time)
	plt.imshow(data)
	plt.colorbar()
	plt.show()


# fig = plt.figure()
# #data = np.random.rand(nx, ny)
# #sns.heatmap(data, vmax=.8, square=True)



# def init():
# 	sns.heatmap(np.zeros((total_pts, total_pts)), vmax=.8, square=True)

# def animate(i):
# 	time = min(hawkes_model.data[:,0])+ (time_length/num_frams)*i
# 	data = func_intensity(time)
# 	sns.heatmap(data, vmax=.8, square=True)

# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frams, repeat = False)
# anim.save('heatmap.mp4')
#plt.show()











