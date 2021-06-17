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
import numpy.ma as ma
 
import warnings
warnings.filterwarnings('ignore')

#============================================================================  
res=np.load('/home/kwu2/res/2021/inversion/0519/dat/output/flux_4.npy')
obs=np.load('/home/kwu2/res/2021/inversion/0519/dat/output/obs_4.npy')
loc=np.load('/home/kwu2/res/2021/inversion/0519/dat/output/loc_4.npy')
sts=np.load('/home/kwu2/res/2021/inversion/0519/dat/output/sts_4.npy')
   
#============================================================================
x, y = np.meshgrid(loc[0,:],loc[1,:])
tem=np.reshape(res,(1200,1200,8))

flux_min = 0
flux_max = 35

nis_min = -5
nis_max = 15

er_min = 0
er_max = 35

fc_min = -15
fc_max = 5

x_1=2.05
x_2=2.6
y_1=48.6
y_2=49.0

#============================================================================
plt.figure(0)   #Truth
plt.contourf(x,y,tem[:,:,0],cmap=cm.jet,levels=np.linspace(flux_min,flux_max,100),extend='both')
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Truth (' + str(round(sts[0,0],2)) + ' tCO$_2$ s$^{-1}$)')
plt.axis([x_1, x_2, y_1, y_2])
plt.show()

plt.figure(1)   #Prior State
plt.contourf(x,y,tem[:,:,1],cmap=cm.jet,levels=np.linspace(flux_min,flux_max,100),extend='both')
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Prior State ('+ str(round(sts[1,0],2)) + '\u00B1' + str(round(sts[1,1],2)) + ' tCO$_2$ s$^{-1}$)')
plt.axis([x_1, x_2, y_1, y_2])
plt.show()

plt.figure(2)   #Posterior State
plt.contourf(x,y,tem[:,:,2],cmap=cm.jet,levels=np.linspace(flux_min,flux_max,100),extend='both')
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Posterior State ('+ str(round(sts[2,0],2)) + '\u00B1' + str(round(sts[2,1],2)) + ' tCO$_2$ s$^{-1}$)')
plt.axis([x_1, x_2, y_1, y_2])
plt.show()

plt.figure(3)   #Flux Correction  
plt.contourf(x,y,tem[:,:,3],cmap=cm.jet,levels=np.linspace(fc_min,fc_max,100),extend='both')
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Flux Correction (' + '\u03BC' + 'mol m$^{-2}$ s$^{-1}$)')
plt.axis([x_1, x_2, y_1, y_2])
plt.show()

plt.figure(4)   #Gain  
plt.contourf(x,y,tem[:,:,4],cmap=cm.jet)
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Gain (' + str(round(np.nanmean(res[:,4]),2)) + ')')
plt.axis([x_1, x_2, y_1, y_2])
plt.show()

plt.figure(5)   #Error Reduction
plt.contourf(x,y,tem[:,:,5],cmap=cm.jet,levels=np.linspace(er_min,er_max,100),extend='both')
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Error Reduction (' + str(round(sts[0,1],2)) + '%)')
plt.axis([x_1, x_2, y_1, y_2])
plt.show()

plt.figure(6)   #Prior Flux Noise
plt.contourf(x,y,tem[:,:,6],cmap=cm.jet,levels=np.linspace(nis_min,nis_max,100),extend='both')
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Prior Flux Noise (' + '\u03BC' + 'mol m$^{-2}$ s$^{-1}$)')
plt.axis([x_1, x_2, y_1, y_2])
plt.show()

plt.figure(7)   #Posterior Flux Noise
plt.contourf(x,y,tem[:,:,7],cmap=cm.jet,levels=np.linspace(nis_min,nis_max,100),extend='both')
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Posterior Flux Noise (' + '\u03BC' + 'mol m$^{-2}$ s$^{-1}$)')
plt.axis([x_1, x_2, y_1, y_2])
plt.show()

plt.figure(8)  #DXCO2 (urban enhancement)     
plt.plot(obs[0,:],'o-',label='Perfect OBS')
plt.plot(obs[3,:],'o-',label='Synthetic OBS')
plt.plot(obs[1,:],'o-',label='Prior State')
plt.plot(obs[4,:],'o-',label='Posterior State')
plt.legend(loc='best')
plt.title('Urban CO$_2$ enhancement')
plt.xlabel('Observation')
plt.ylabel('DX CO$_2$ (ppm)')
plt.show()

wk=ma.corrcoef(ma.masked_invalid(res[:,3]), ma.masked_invalid(res[:,6]))
plt.figure(9) #flux correction vs prior flux noise   
plt.plot(res[:,3],res[:,6],'o')
plt.title('Correlation Coefficient = ' + str(round(wk[0,1],2)))
plt.xlabel('Flux correction (' + '\u03BC' + 'mol m$^{-2}$ s$^{-1}$)')
plt.ylabel('Prior flux noise (' + '\u03BC' + 'mol m$^{-2}$ s$^{-1}$)')
plt.show()


#plt.savefig('D:\\Edin\\2021\\Inversion\\0118\\pic\\0_9.png')