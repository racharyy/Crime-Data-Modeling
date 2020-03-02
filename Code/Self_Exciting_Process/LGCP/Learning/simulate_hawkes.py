from hawkes_synth import *


mu = 0.5
theta =0
sigma = 1
a = 0.2
b=0.4
N = 100


obj = Hawkes(mu,a,sigma,theta,b,N)
obj.generate_data()
obj.plot_data()