#from sklearn.cluster import KMeans
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import bernoulli
import re
import csv
import pickle as pk
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from sklearn import metrics
import datetime as dt


def create_date(date):
	date=date.split('-')
	return dt.date(int(date[0]),int(date[1]),int(date[2]))


def min_date(date_list):
	temp_min = dt.date.today()
	for date in date_list:
		date = create_date(date)
		if (date - temp_min).days < 0:
			temp_min = date

	return temp_min


def max_date(date_list):
	temp_max = dt.date.min
	for date in date_list:
		date = create_date(date)
		if (date - temp_max).days > 0:
			temp_max = date
			
	return temp_max


num_timesteps = 200



#Process the data and get the space and time of the crime
with open('../../Data/prop.csv') as fin:
    with open('../../Data/AR_data/spatio_temp.csv','wb') as fout:
        reader = csv.DictReader(fin)
        fieldnames = ['LONGITUDE','LATITUDE','DATE_ON']
        writer = csv.DictWriter(fout,fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
        	if (row['LONGITUDE'] and row['LATITUDE'] and row['DATE_ON']):
        		write_dict={}
        		write_dict['LONGITUDE']= row['LONGITUDE']
        		write_dict['LATITUDE'] = row['LATITUDE']
        		write_dict['DATE_ON'] = row['DATE_ON']

        		writer.writerow(write_dict)



#Save the spatio temporal data in a dictionary
with open('../../Data/AR_data/spatio_temp.csv','rb') as fin:
 	reader = csv.DictReader(fin)
 	spatial_list =[]
 	for row in reader:	
 		spatial_list.append([row['LONGITUDE'],row['LATITUDE']])


pk.dump(np.array(spatial_list) ,open( "../../Data/AR_data/spatial_list.p", "wb"))
relevant_data=pk.load(open('../../Data/AR_data/spatial_list.p','rb'))


for num_cluster in range(10,11):
	estimator=KMeans(init='k-means++', n_clusters=num_cluster, n_init=10)
	estimator=estimator.fit(relevant_data)

	# print("silhouette_score:",metrics.silhouette_score(relevant_data, estimator.labels_,metric='euclidean',sample_size=len(relevant_data)))
	colors = cm.spectral(estimator.labels_.astype(float) /num_cluster)
	plt.scatter(relevant_data[:, 0], relevant_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,c=colors)
	# Labeling the clusters
	centers = estimator.cluster_centers_
	labels = estimator.labels_
	#Transforming data to cluster centers
	ind = 0
	with open('../../Data/AR_data/spatio_temp.csv','rb') as fin:
 		reader = csv.DictReader(fin)
 		hotspot_dict = {}
 		for row in reader:
	 		center = centers[labels[ind]]
	 		center =(center[0],center[1]) 
	 		try:
	 			hotspot_dict[center].append(row['DATE_ON'])
	 		except KeyError:
	 			hotspot_dict[center]=[row['DATE_ON']]
	 		ind=ind+1

	for keys in hotspot_dict.keys():
 		occurences = hotspot_dict[keys]
 		occurences.sort()
 		hotspot_dict[keys] = occurences

	pk.dump(hotspot_dict,open( "../../Data/AR_data/hotspot_dict.p", "wb"))
 	hotspot_dict = pk.load(open( "../../Data/AR_data/hotspot_dict.p", "rb"))	

 	#Creating the time series data of the hotspots from 
 	first_date = min_date([hotspot_dict[key][0] for key in hotspot_dict.keys()])
 	last_date = max_date([hotspot_dict[key][-1] for key in hotspot_dict.keys()])

 	total_days = (last_date-first_date).days

 	data_dimension = num_cluster
 	num_timesteps = total_days+1
 	AR_data = np.zeros((data_dimension,num_timesteps))
 	for ind,dim in enumerate(hotspot_dict.keys()):
 		dates_for_currentdim = hotspot_dict[dim]
 		for date in dates_for_currentdim:
 			cur_date = create_date(date)
 			pos = (cur_date-first_date).days
 			AR_data[ind][pos] = AR_data[ind][pos]+1


 	
 	pk.dump(AR_data,open("../../Data/AR_data/AR_data.p", "wb"))


 	#Save the processed count data in CSV file
 	AR_data = pk.load(open("../../Data/AR_data/AR_data.p", "rb"))
 	with open('../../Data/AR_data/hotspot_crime_count.csv','wb') as fout:
 		fieldnames =['DATE_ON']+ [str('hotspot ')+str(i) for i in range(1,data_dimension+1)]
 		writer = csv.DictWriter(fout,fieldnames=fieldnames)
 		writer.writeheader()
 		date = first_date
 		date_ind=0
 		while (last_date-date).days>=0:
 			
 			write_dict={}
 			cur_date=date.strftime('%Y-%m-%d')
 			write_dict['DATE_ON'] = cur_date
 			
 			for dim in xrange(1,data_dimension+1):
 				write_dict[str('hotspot ')+str(dim)]=AR_data[dim-1][date_ind]
 			

 			writer.writerow(write_dict)
 			date += dt.timedelta(days=1)
 			date_ind=date_ind+1




	# Draw white circles at cluster centers
	plt.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200)

	for i, c in enumerate(centers):
	    plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

	plt.ylabel("LONGITUDE")
	plt.xlabel("LATITUDE")
	img_name="cluster_crime_"+str(num_cluster)
	plt.savefig(img_name)
	#plt.close()
	plt.show()
	silhouette_score=metrics.silhouette_score(relevant_data, estimator.labels_,metric='euclidean',sample_size=15000)
	row=[num_cluster,silhouette_score,estimator.cluster_centers_.tolist()] 
	#print(silhouette_score)
	#write_csv(row)
