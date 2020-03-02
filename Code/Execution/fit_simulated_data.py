import numpy as np 
import matplotlib.pyplot as plt 
from generate_synth import *
import sys
sys.path
sys.path.append('./')
from Self_Exciting_Process.space_time_Hawkes.learning.hawkes_EM import *



#Generate synthetic data
num_frames = 10
given_range,cell_division,max_intensity,T = [[0,2],[0,2]],10,2,10
data,sthp = generate_hawkes(given_range,cell_division+1,max_intensity,T,num_frames,show_plot=False)
true_param = sthp.params


#Fit generated synthetic data
background_num=cell_division
hawkes_model = Estimation_hawkes(data,background_num,true_param)
estimated_param=hawkes_model.estimate_params()


x_axis = range(5)
x_label=['K','time_offset','dist_offset','time_power','dist_power']

true_param_val = [true_param.K,true_param.time_offset,true_param.dist_offset,true_param.time_power,true_param.dist_power]
estimated_param_val = [estimated_param.K,estimated_param.time_offset,estimated_param.dist_offset,estimated_param.time_power,estimated_param.dist_power]



plt.plot(x_axis,true_param_val,'ro')
plt.plot(x_axis,estimated_param_val,'bo')
plt.title('True parameters (in Red) vs Estimated Parametrs (in Blue)')



plt.xticks(x_axis, x_label, rotation='vertical')
plt.savefig('true vs learned parameters.png')
plt.show()
plt.close()


fig = plt.figure()
x = np.arange(background_num-1)
y = np.arange(background_num-1)
X,Y = np.meshgrid(x,y)
#print true_param.background.shape,estimated_param.background.shape
Z = abs(true_param.background - estimated_param.background)
plt.imshow(Z)
plt.colorbar()
plt.show()
