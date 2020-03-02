import mysql.connector

cnx = mysql.connector.connect(user="root", password="GIDSRDSC", host="127.0.0.1", database="mysql")
#cnx = mysql.connector.connect(user="root", password="GIDSRDSC", host="127.0.0.1", database="TASS_Iron")
cursor = cnx.cursor()
query = open("query.txt", "r")  # in
# select(HEX(track_uuid)) as TRACKID ,
# target_latitude, target_longitude, timestamp_epoch_us
# from track_points order by TRACKID, timestamp_epoch_us


cursor.execute(query.readline())

results = cursor.fetchall()

cnx.close()

import simplekml
#kml = simplekml.Kml()
#kml.newlinestring(name="Pathway", coords=[lon,lat])
#lin = kml.newlinestring(name="Pathway", coords=[(t[1],t[2]) for t in results if t[0] == results[0][0]])
#lin.style.linestyle.color = simplekml.Color.yellowgreen
#lin.style.linestyle.width = 20
#kml.save("lin.kml")
#print(results)

tracks = set([t[0] for t in results])
kml = simplekml.Kml()
multiline = kml.newmultigeometry(name="MultiLine")
multiline.style.linestyle.color = simplekml.Color.red
multiline.style.linestyle.width = 15
for tr in tracks:
        multiline.newlinestring(name="Pathway", coords=[(t[1],t[2]) for t in results if t[0] == tr])

kml.save("Tracks.kml")

# del(sys.modules["matplotlib"]) # use this to clear the module
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import smopy

# bounding box
minlon = min(t[1] for t in results)
maxlon = max(t[1] for t in results)

minlat = min(t[2] for t in results)
maxlat = max(t[2] for t in results)

'''
map = smopy.Map((minlat,minlon,maxlat,maxlon),z = 15)
lat = [t[2] for t in results if t[0] == '01A9C816A1C94ACEA1E3C643ACC716BF']
lon = [t[1] for t in results if t[0] == '01A9C816A1C94ACEA1E3C643ACC716BF']
#for i in range(len(lat)):
x,y = map.to_pixels(np.array(lat),np.array(lon))
plt.plot(x, y, '-r', linewidth=1) # r = 'red'
ax = map.show_mpl(figsize = (8,6), dpi = 72) # show the image in matplotlib
ax.plot(x,y)


plt.show()

plt.savefig("hoyhoy.png") # writes the file
map.save_png("hoy.png") # write the tile to png
'''

# use folium
import folium
import datetime
# lat , lon lools like
map_osm = folium.Map(location=[(minlat+maxlat)/2, (minlon+maxlon)/2], zoom_start=14)



for id in set([id[0] for id in results]):
    lat = [t[2] for t in results if t[0] == id]
    lon = [t[1] for t in results if t[0] == id]
    points = []
    for ii in (0,len(lat)-1):
        points.append(tuple([lat[ii], lon[ii]]))

    folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(map_osm)

map_osm.save('map.html')

# map trip endpoints
map_endpoints = folium.Map(location=[(minlat+maxlat)/2, (minlon+maxlon)/2], zoom_start=14)

endpoints = list()
endpointtimes = list()
for id in set([id[0] for id in results]):
    lat = [t[2] for t in results if t[0] == id]
    lon = [t[1] for t in results if t[0] == id]
    time = [t[3]/1000000 for t in results if t[0] == id]
    endpoints.append(tuple([lat[-1], lon[-1]]))
    endpointtimes.append(time[-1])
for each in endpoints:
    folium.CircleMarker(each, radius=1).add_to(map_endpoints)

map_endpoints.save('map_endpoints.html')

endtimes = list()
for t in endpointtimes:
    endtimes.append(datetime.datetime.fromtimestamp(t))


hours = [t.hour for t in endtimes]

seconds = [t.second for t in endtimes]


# output endpoints in KML
# count endpoints
uTrackIds = list(set([id[0] for id in results]))
nUniqueTracks = len(uTrackIds)
kmlend = simplekml.Kml()
endpoints = list()
count = 0

for id in uTrackIds[0:30]:
    track = [t for t in results if t[0] == uTrackIds[0]]
    #lat = [t[2] for t in results if t[0] == id]
    #lon = [t[1] for t in results if t[0] == id]

    #endpoints.append(tuple([lat[-1], lon[-1]]))
    endpoints.append(tuple([track[-1][2], track[-1][1]]))
    pnt = kmlend.newpoint()
    pnt.stylemap.normalstyle.labelstyle.color = simplekml.Color.blue
    #pnt.stylemap.highlightstyle.labelstyle.color = simplekml.Color.red
    pnt.coords = [(lon[-1], lat[-1])]
    count = count+1
    if count % 10 == 0:
        print(count/nUniqueTracks)


kmlend.save("EndPoints.kml")


# don't forget to look at 'staypoints' for these trajectories (if they exist -- tracking could stop when vehicle stops)

# use endpoints since they are likely to to simply come from out of view into the scene
#writing Irondequoit endpoints
cnx = mysql.connector.connect(user="root", password="GIDSRDSC", host="127.0.0.1", database="TASS_Iron")
cursor = cnx.cursor()
query = open("query.txt", "r")  # in
# select(HEX(track_uuid)) as TRACKID ,
# target_latitude, target_longitude, timestamp_epoch_us
# from track_points order by TRACKID, timestamp_epoch_us


cursor.execute(query.readline())

results = cursor.fetchall()

cnx.close()
endpoints = list()
endpointtimes = list()

for id in set([id[0] for id in results]):
    lat = [t[2] for t in results if t[0] == id]
    lon = [t[1] for t in results if t[0] == id]
    time = [t[3]/1000000 for t in results if t[0] == id]
    endpoints.append(tuple([lat[-1], lon[-1]]))
    endpointtimes.append(time[-1])


import csv
with open('Irondequoit_endpoints', 'w',newline = '')



