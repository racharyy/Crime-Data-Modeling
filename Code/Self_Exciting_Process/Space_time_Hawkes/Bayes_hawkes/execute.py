import matplotlib.pyplot as pyplot
import numpy as np 
from scipy.stats import norm 
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from pyhmc import hmc
from hawkes_bayes import *
import csv





data=np.loadtxt(open("../../../../Data/elmgrove_end_all.csv", "rb"), delimiter=",", skiprows=1)[:100]
Bayes_model = Bayes_Hawkes(data)
Bayes_model.fit_background()
Bayes_model.posterior_sampling()
