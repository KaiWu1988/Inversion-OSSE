#=================plot different OBS unc====================================
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
from sklearn.preprocessing import scale
 
import warnings
warnings.filterwarnings('ignore')

dat_1 = np.load('/home/kwu2/res/2021/inversion/0517/res_1.npy')
dat_2 = np.load('/home/kwu2/res/2021/inversion/0517/res_2.npy')
dat_3 = np.load('/home/kwu2/res/2021/inversion/0517/res_3.npy')
dat_4 = np.load('/home/kwu2/res/2021/inversion/0517/res_4.npy')


plt.figure(0)       
plt.plot(dat_1[0,:],dat_1[3,:],'o-',label='MicroCarb-All Data (366)')
plt.plot(dat_2[0,:],dat_2[3,:],'o-',label='MicroCarb-Cloud Free (225)')
plt.plot(dat_3[0,:],dat_3[3,:],'o-',label='OCO$_3$-All Data (1500)')
plt.plot(dat_4[0,:],dat_4[3,:],'o-',label='OCO$_3$-Cloud Free (895)')
plt.legend(loc='best')
plt.xlabel('Observation Uncertainty (ppm)')
plt.ylabel('Error Reduction (%)')
plt.show()

plt.figure(1)       
plt.plot(dat_1[0,:],dat_1[4,:],'o-',label='MicroCarb-All Data (366)')
plt.plot(dat_2[0,:],dat_2[4,:],'o-',label='MicroCarb-Cloud Free (225)')
plt.plot(dat_3[0,:],dat_3[4,:],'o-',label='OCO$_3$-All Data (1500)')
plt.plot(dat_4[0,:],dat_4[4,:],'o-',label='OCO$_3$-Cloud Free (895)')
plt.legend(loc='best')
plt.xlabel('Observation Uncertainty (ppm)')
plt.ylabel('Gain')
plt.show()

plt.figure(2)       
plt.plot(dat_1[0,:],dat_1[1,:],'o-',label='MicroCarb-All Data (366)')
plt.plot(dat_2[0,:],dat_2[1,:],'o-',label='MicroCarb-Cloud Free (225)')
plt.plot(dat_3[0,:],dat_3[1,:],'o-',label='OCO$_3$-All Data (1500)')
plt.plot(dat_4[0,:],dat_4[1,:],'o-',label='OCO$_3$-Cloud Free (895)')
plt.legend(loc='best')
plt.xlabel('Observation Uncertainty (ppm)')
plt.ylabel('Posterior Total Emissions (tCO$_2$ s$^{-1}$)')
plt.show()

plt.figure(3)       
plt.plot(dat_1[0,:],dat_1[2,:],'o-',label='MicroCarb-All Data (366)')
plt.plot(dat_2[0,:],dat_2[2,:],'o-',label='MicroCarb-Cloud Free (225)')
plt.plot(dat_3[0,:],dat_3[2,:],'o-',label='OCO$_3$-All Data (1500)')
plt.plot(dat_4[0,:],dat_4[2,:],'o-',label='OCO$_3$-Cloud Free (895)')
plt.legend(loc='best')
plt.xlabel('Observation Uncertainty (ppm)')
plt.ylabel('Posterior Flux Bias (tCO$_2$ s$^{-1}$)')
plt.show()