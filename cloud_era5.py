#=================define cloud free data=============================



import csv
import numpy as np
import netCDF4 as nc
import pandas as pd
import random
import matplotlib as mpl
import pylab as plt
from matplotlib import cm
from scipy.io import loadmat
from scipy import array, linalg, dot
from haversine import haversine, Unit
from itertools import islice
from sklearn.preprocessing import scale
from scipy import interpolate
from scipy.interpolate import NearestNDInterpolator

import warnings
warnings.filterwarnings('ignore')

#==================read cloud========================================
filename_1='/home/kwu2/res/2021/inversion/0329/dat/era5/cloud_london.nc'

loc = []
time= []
z_1 = []
file_obj = nc.Dataset(filename_1)                    #read tcc
loc.append(file_obj.variables['longitude'][:])       #lon
loc.append(file_obj.variables['latitude'][:])        #lat
time.append(file_obj.variables['time'][:])
z_1.append(file_obj.variables['tcc'][:])             #tcc  
file_obj.close()

loc = np.array(loc)
time= np.array(time)
z_1 = np.array(z_1)

time = 0

z_2 = z_1[0,time,:,:]                        #2020/04/12  12:00

nx=len(loc[0,:])
ny=len(loc[1,:])
num_p=nx*ny

x, y = np.meshgrid(loc[0,:],loc[1,:])
z = np.full((num_p,3), np.nan)  

n = 0
for i in range(nx):
    for j in range(ny):
        z[n,0]=y[i,j]
        z[n,1]=x[i,j]
        z[n,2]=z_2[i,j]
        n=n+1

#==================read mc==========================================  
filename_2='/home/kwu2/res/2021/inversion/0519/dat/input/loc_mc_l.csv'

loc_tem = []
with open(filename_2) as csvfile:
      reader = csv.reader(csvfile)
      for row in islice(reader, 1, None):      
          loc_tem.append(row)         
loc_tem = np.array(loc_tem)
loc_tem = loc_tem.astype(float)
nobs=len(loc_tem)

#===================================================================
f = NearestNDInterpolator(list(zip(z[:,1],z[:,0])), z[:,2])

z_3 = np.full((nobs,5), np.nan)  
z_3[:,0:2]=loc_tem

for i in range(nobs):
    z_3[i,2] = f(z_3[i,1], z_3[i,0])
    
s=2.0
Fps=(26.098*(s**(-0.45))+10.18)/(26.098+10.18)

g=0.8
dat_1 = []
dat_2 = []
for i in range(nobs):
    z_3[i,3] = Fps*g*(1-z_3[i,2])
    rc = random.uniform(0,1)
    if rc<z_3[i,3]:
        z_3[i,4]=0
        dat_1.append(z_3[i,:])
    else:
        z_3[i,4]=1
        dat_2.append(z_3[i,:])

dat_1 = np.array(dat_1)
dat_2 = np.array(dat_2)
nobs_1= len(dat_1)
nobs_2= len(dat_2)

with open('/home/kwu2/res/2021/inversion/0418/dat/input/loc_mc_london.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(z_3)

#==================plot figure=======================================
# lat = 48.8566
# lon = 2.3522

lat = 51.5074
lon = -0.1278

tit='MicroCarb city-mode sampling pattern London'
#tit='OCO-3 SAM sampling pattern Paris'

plt.plot(dat_1[:,1],dat_1[:,0],'ro',markersize=1,label='QF=0')
plt.plot(dat_2[:,1],dat_2[:,0],'bo',markersize=1,label='QF=1')
plt.plot(lon,lat,'r*',markersize=10)        
plt.legend(loc='best')
plt.title(tit)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

tit='ERA5 cloud fraction APR 12 12:00 2020 London'
x, y = np.meshgrid(loc[0,:],loc[1,:])
cflux = plt.contourf(x,y,z_2,cmap=cm.jet)     #odiac                            
plt.colorbar()
plt.title(tit)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()