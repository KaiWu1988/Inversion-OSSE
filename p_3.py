#=================test different OBS unc====================================
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

#============================================================================  
sts_1=np.load('/home/kwu2/res/2021/inversion/0517/dat_1/sts_1.npy')
sts_2=np.load('/home/kwu2/res/2021/inversion/0517/dat_1/sts_2.npy')
sts_3=np.load('/home/kwu2/res/2021/inversion/0517/dat_1/sts_3.npy')
sts_4=np.load('/home/kwu2/res/2021/inversion/0517/dat_1/sts_4.npy')
sts_5=np.load('/home/kwu2/res/2021/inversion/0517/dat_1/sts_5.npy')
sts_6=np.load('/home/kwu2/res/2021/inversion/0517/dat_1/sts_6.npy')
sts_7=np.load('/home/kwu2/res/2021/inversion/0517/dat_1/sts_7.npy')

res_1=np.load('/home/kwu2/res/2021/inversion/0517/dat_1/flux_1.npy')
res_2=np.load('/home/kwu2/res/2021/inversion/0517/dat_1/flux_2.npy')
res_3=np.load('/home/kwu2/res/2021/inversion/0517/dat_1/flux_3.npy')
res_4=np.load('/home/kwu2/res/2021/inversion/0517/dat_1/flux_4.npy')
res_5=np.load('/home/kwu2/res/2021/inversion/0517/dat_1/flux_5.npy')
res_6=np.load('/home/kwu2/res/2021/inversion/0517/dat_1/flux_6.npy')
res_7=np.load('/home/kwu2/res/2021/inversion/0517/dat_1/flux_7.npy')

tem=np.full((5,7), np.nan) 

tem[0,0]=sts_1[1,3]
tem[1,0]=sts_1[2,0]
tem[2,0]=sts_1[2,0]-sts_1[0,0]
tem[3,0]=sts_1[0,1]
tem[4,0]=np.nanmean(res_1[:,4])

tem[0,1]=sts_2[1,3]
tem[1,1]=sts_2[2,0]
tem[2,1]=sts_2[2,0]-sts_2[0,0]
tem[3,1]=sts_2[0,1]
tem[4,1]=np.nanmean(res_2[:,4])

tem[0,2]=sts_3[1,3]
tem[1,2]=sts_3[2,0]
tem[2,2]=sts_3[2,0]-sts_3[0,0]
tem[3,2]=sts_3[0,1]
tem[4,2]=np.nanmean(res_3[:,4])

tem[0,3]=sts_4[1,3]
tem[1,3]=sts_4[2,0]
tem[2,3]=sts_4[2,0]-sts_4[0,0]
tem[3,3]=sts_4[0,1]
tem[4,3]=np.nanmean(res_4[:,4])

tem[0,4]=sts_5[1,3]
tem[1,4]=sts_5[2,0]
tem[2,4]=sts_5[2,0]-sts_5[0,0]
tem[3,4]=sts_5[0,1]
tem[4,4]=np.nanmean(res_5[:,4])

tem[0,5]=sts_6[1,3]
tem[1,5]=sts_6[2,0]
tem[2,5]=sts_6[2,0]-sts_6[0,0]
tem[3,5]=sts_6[0,1]
tem[4,5]=np.nanmean(res_6[:,4])

tem[0,6]=sts_7[1,3]
tem[1,6]=sts_7[2,0]
tem[2,6]=sts_7[2,0]-sts_7[0,0]
tem[3,6]=sts_7[0,1]
tem[4,6]=np.nanmean(res_7[:,4])

plt.figure(0)       
plt.plot(tem[0,:],tem[3,:],'o-')
plt.xlabel('Observation Uncertainty (ppm)')
plt.ylabel('Error Reduction (%)')
plt.show()

plt.figure(1)       
plt.plot(tem[0,:],tem[4,:],'o-')
plt.xlabel('Observation Uncertainty (ppm)')
plt.ylabel('Gain')
plt.show()

plt.figure(2)       
plt.plot(tem[0,:],tem[2,:],'o-')
plt.xlabel('Observation Uncertainty (ppm)')
plt.ylabel('Posterior Uncertainty')
plt.show()

plt.figure(3)       
plt.plot(tem[0,:],tem[1,:],'o-')
plt.xlabel('Observation Uncertainty (ppm)')
plt.ylabel('Posterior total emissions')
plt.show()

np.save('/home/kwu2/res/2021/inversion/0517/res_1.npy',tem)
