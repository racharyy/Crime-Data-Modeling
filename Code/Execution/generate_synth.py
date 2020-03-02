import numpy as np 
import matplotlib.pyplot as plt 
import sys
sys.path
sys.path.append('./')
from Self_Exciting_Process.space_time_Hawkes.simulation.hawkes_simulate import *
import matplotlib.animation as anim
import pandas as pd
from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from math import floor



def generate_hawkes(given_range,cell_division,max_intensity,T,num_frames,show_plot=True):
	
	sthp = Spatio_temp_hawkes(given_range,cell_division,max_intensity,T)
	sthp.simulate()
	data=np.array(sthp.data)

	if show_plot:

		dic_list=[{'x_axis':[],'y_axis':[]} for frame in range(num_frames)]

		for point in data:
			frame_diff =int(T/num_frames)
			ind = int(point[0]/frame_diff)
			#print ind
			dic_list[ind]['x_axis'].append(point[1])
			dic_list[ind]['y_axis'].append(point[2])

		frame_list = [DataFrame(dic) for dic in dic_list]

		# First set up the figure, the axis, and the plot element we want to animate
		fig = plt.figure()
		plt.grid(True)
		ax = plt.axes(xlim=(0, 100), ylim=(0, 100))
		line, = ax.plot([], [], 'bo', ms=6)	


		# initialization function: plot the background of each frame
		def init():
		    line.set_data([], [])
		    return line,

		# animation function of dataframes' list
		def animate(i):
		    line.set_data(frame_list[i]['x_axis'], frame_list[i]['y_axis'])
		    return line,

		# call the animator, animate every 300 ms
		# set number of frames to the length of your list of dataframes
		anim = animation.FuncAnimation(fig, animate, frames=num_frames, init_func=init, interval=300, blit=True)
		anim.save('../Self_Exciting_Process/space_time_Hawkes/Plots/space_time_Hawkes_simulation.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

		plt.show()

	return data,sthp





# threedee = plt.figure().gca(projection='3d')
# threedee.scatter(time_ar,x_ar,y_ar)
# threedee.set_xlabel('x_axis')
# threedee.set_ylabel('y_axis')
# threedee.set_zlabel('Time')
# plt.show()

