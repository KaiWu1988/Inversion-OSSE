#=================inversion OSSE======================================
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

import warnings
warnings.filterwarnings('ignore')

#=================parameter setting==================================
obs_bias=0                       # obs bias
obs_rerr=1.0                     # obs random error

flux_bias=3                      # flux bias
flux_rerr=5                      # flux random error
correl_flag=10                   # spatial correlation length
nis_sample=2                     # flux noise sample

# #==================read odiac========================================
filename_1='/home/kwu2/res/2021/inversion/0418/dat/input/odiac2019_1kmx1km_201804_Paris.nc'

loc = []
z_1 = []
file_obj = nc.Dataset(filename_1)               #read odiac
loc.append(file_obj.variables['lon'][:])        #lon
loc.append(file_obj.variables['lat'][:])        #lat
z_1.append(file_obj.variables['emiss'][:])      #odiac  umol m-2 s-1
file_obj.close()

loc = np.array(loc)
z_1 = np.array(z_1)

nx=len(loc[0,:])
ny=len(loc[1,:])
num_p=nx*ny

X, Y = np.meshgrid(loc[0,:], loc[1,:])
X = X.flatten()
Y = Y.flatten()

#==================read lat/lon of obs================================
filename_2='/home/kwu2/res/2021/inversion/0418/dat/input/loc_mc_1.csv'

loc_tem = []
with open(filename_2) as csvfile:
      reader = csv.reader(csvfile)
      for row in islice(reader, 0, None):      
          loc_tem.append(row)         
loc_tem = np.array(loc_tem)
loc_tem = loc_tem.astype(float)

#==========all data==================
# loc_1 = loc_tem
# nobs=len(loc_1) 

#==========cloud free================
loc_1 = []
for i in range(len(loc_tem)):
    if loc_tem[i,4]==0:                  #oco 2 mc 4
        loc_1.append(loc_tem[i,:])
loc_1 = np.array(loc_1)
nobs=len(loc_1)   

#=========attach filename============
fl = []
for k in range(0,nobs):
    aa = str(loc_1[k,1])
    bb = str(loc_1[k,0])

    if aa[-1]=='0':
        aa = str(int(loc_1[k,1]))

    if bb[-1]=='0':
        bb = str(int(loc_1[k,0]))

    fl.append('202004121100'+'_'+aa+'_'+bb+'_X_foot.nc')

# #==================read footprint====================================
filename_3='/exports/csce/datastore/geos/users/kwu2/output_mc/out_2020041211_gfs0p25_ideal/footprints/'      

z_2 = []  
for k in range(nobs):
    str1 = ''.join(fl[k]).strip('\n')
    filename = filename_3+str1
    file_obj = nc.Dataset(filename)
    z_2.append(np.flipud(np.squeeze(file_obj.variables['foot'][:])))  #footprint  ppm/umol m-2 s-1
    file_obj.close()    
z_2 = np.array(z_2)

# # # np.save('/exports/csce/datastore/geos/users/kwu2/H_4.npy',z_2)    #save footprint

# z_2=[]
# z_2=np.load('/exports/csce/datastore/geos/users/kwu2/H_4.npy')        #read footprint

# ##==================define city area=======================================
x_1 = 2.05
y_1 = 48.6

x_2 = 2.6
y_2 = 49

city_1=[x_1, x_2, x_2, x_1, x_1]
city_2=[y_1, y_1, y_2, y_2, y_1]

##===================plot figure=======================================
plt.figure(1)
x, y = np.meshgrid(loc[0,:],loc[1,:])
plt.contourf(x,y,np.log10(z_1[0,:,:]),cmap=cm.jet)      #odiac                            
plt.colorbar()
plt.plot(loc_1[:,1],loc_1[:,0],'ro')                    #obs
#plt.plot(city_1,city_2,'r')
#plt.contourf(x,y,np.log10(z_2[0,:,:]),cmap=cm.jet)     #footprint
#plt.colorbar()
plt.axis([1.9, 2.8, 48.4, 49.2])
# plt.axis([1.5, 3.5, 46.4, 49.2])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.show()

##================reshape arrays to vectors for calculation===========          
z_3 = np.full((num_p,1), np.nan)  
z_3[:,0] = z_1[0,:,:].flatten()                  #odiac
del z_1

z_4 = np.full((num_p,nobs), np.nan)
z_5 = np.full((num_p,nobs), np.nan)   

for k in range(nobs):
    z_4[:,k] = z_2[k,:,:].flatten()           #footprint
    z_5[:,k] = z_4[:,k]*z_3[:,0]              #xco2

del z_2  

#=======index of noneffective pixels (not in the city area or no footprint)===========
#=======index of effective pixels (in the city area and covered by footprint)=========
inx_1 = []
inx_2 = []

for i in range(num_p):
    if x_1<X[i,]<x_2 and y_1<Y[i,]<y_2:
        t=0
        for k in range(nobs):
            if z_5[i,k]!=0:
                inx_2.append(i)
                t=1
                break;
        if t==0:
            inx_1.append(i)
    else:
        inx_1.append(i)

nflux = len(inx_2) 
print(nflux)

##=================================extract effective pixels===========================
dat=np.full((nflux,2*nobs+8), np.nan) 

dat[:,0]=z_3[inx_2,0]                   #odiac
dat[:,1:nobs+1]=z_4[inx_2,:]            #footprint
dat[:,nobs+1:2*nobs+1]=z_5[inx_2,:]     #xco2
del z_3, z_4, z_5

# np.save('/home/kwu2/res/2021/inversion/0517/dat_4.npy', dat)          
# np.save('/home/kwu2/res/2021/inversion/0517/inx_1.npy', inx_1)          
# np.save('/home/kwu2/res/2021/inversion/0517/inx_2.npy', inx_2)          

# #====================================================================================
# dat=np.load('/home/kwu2/res/2021/inversion/0517/dat_4.npy')
# inx_1=np.load('/home/kwu2/res/2021/inversion/0517/inx_1.npy') 
# inx_2=np.load('/home/kwu2/res/2021/inversion/0517/inx_2.npy') 
# nflux = len(inx_2) 
# nobs=895

#=================================extract effective lat/lon==========================
loc_2 = np.full((len(inx_2),2), np.nan) 
loc_2[:,0] = X[inx_2,]
loc_2[:,1] = Y[inx_2,]
del X, Y

#=============remove hot spots===========================================     
kk = np.where(dat[:,0]>30)[0]

hs = np.full((len(kk),3), np.nan) 
for i in range(len(kk)):
    hs[i,0] = dat[kk[i],0]
    hs[i,1] = loc_2[kk[i],0]
    hs[i,2] = loc_2[kk[i],1]
    dat[kk[i],0] = dat[kk[i]-1,0]
    
#plt.hist(dat[:,0])
#plt.plot(dat[:,0])

#============define flux unc matrix (B)====================================== 
filename_4='/home/kwu2/res/2021/inversion/0516/dat/input/fluxunc_'+str(flux_bias)+'_'+str(flux_rerr)+'_'+str(correl_flag)+'.npy'
B = np.load(filename_4)                                 #read B matrix

# # #==========================================================================
fns = np.ones(nflux)*(flux_bias**(2)+flux_rerr**(2))    
B = np.diag(fns) 

if correl_flag>0:
    for i in range(nflux):                                 
        for j in range(i+1,nflux):
            dis = haversine((loc_2[i,1],loc_2[i,0]),(loc_2[j,1],loc_2[j,0]))  #distance, km
            expon=dis/correl_flag
            B[i,j]=np.sqrt(B[i,i]*B[j,j])*np.exp(-expon)
            B[j,i]=B[i,j] 
np.save(filename_4, B)
 
#====define flux noise from the B matrix (eigenvalue decomposition)============
C = linalg.cholesky(B,lower=True)                   # cholesky decomposition   B = C*C'

np.random.seed(0)                                                # seed of random flux noise
flux_nis = np.random.normal(flux_bias, flux_rerr, (nflux,10))    # 10 normalized flux noise

for j in range(10):
    flux_nis[:,j] = flux_nis[:,j]*(0.5*dat[:,0])

D = np.dot(C,flux_nis)                              # link spatial correlation with flux noise

flux_noise = D[:,nis_sample]                        # choose one of the flux noise
                                   
flux_noise_1 = (flux_noise-flux_noise.mean())/flux_noise.std()                 # normalize the noise
flux_noise_2 = flux_noise_1*flux_rerr+flux_bias     # add random error and bias to the choosed noise

dat[:,2*nobs+1] = flux_noise_2                      # prior noise
dat[:,2*nobs+2] = dat[:,0] + dat[:,2*nobs+1]        # prior state

#=====================synthetic OBS======================================
obs=np.full((5,nobs), np.nan) 

for j in range(nobs):
    obs[0,j] = sum(dat[:,0]*dat[:,1+j])             # perfect DXCO2
    obs[1,j] = sum(dat[:,2*nobs+2]*dat[:,1+j])      # modeled DXCO2

fns = np.ones(nobs)*(obs_bias**(2)+obs_rerr**(2))   # define R matrix
R = np.diag(fns)   

np.random.seed(0)                                  # seed of random flux noise
obs_noise_tem = np.random.normal(obs_bias, obs_rerr, nobs)   # obs noise
obs_noise = (obs_noise_tem-obs_noise_tem.mean())/obs_noise_tem.std()           # normalize the noise

obs[2,:] = obs_noise*obs_rerr+obs_bias             # obs noise
obs[3,:] = obs[0,:]+obs[2,:]                       # synthetic data

#=====================plot data============================================
# plt.plot(obs[0,700:750],'-',label='Perfect OBS')
# plt.plot(obs[1,700:750],'-',label='Prior State')
# plt.plot(obs[2,700:750],'-',label='OBS noise')
# plt.plot(obs[3,700:750],'-',label='Synthetic OBS')
# plt.legend(loc='best')
# plt.title('Urban CO$_2$ enhancement')
# plt.xlabel('Observation')
# plt.ylabel('DX CO$_2$ (ppm)')
# plt.show()

#==================define the H matrix======================================   
H=np.full((nobs,nflux), np.nan) 
for k in range(nobs):
    H[k,:] = np.transpose(dat[:,1+k])

#==================define the prior state===================================  
x0 = dat[:,2*nobs+2]

#===============inversion calculations======================================
tem_1=np.dot(H,B)                    # HB
tem_2=np.transpose(tem_1)            # (HB)'
tem_3=np.dot(tem_1,np.transpose(H))  # HBH'
tem_4=np.linalg.inv(tem_3+R)         # inv(HBH'+R)
K=np.dot(tem_2,tem_4)                # K

dat[:,2*nobs+3]=x0+np.dot(K,np.transpose(obs[3,:]-obs[1,:]))   # posterior state
A=B-np.dot(K,tem_1)                                            # posterior unc.

#===============inversed results estimation=================================   
dat[:,2*nobs+4]=dat[:,2*nobs+3]-dat[:,2*nobs+2]                                   # flux correction
dat[:,2*nobs+5]=1-abs(dat[:,2*nobs+3]-dat[:,0])/abs(dat[:,2*nobs+2]-dat[:,0])     # gain    
dat[dat[:,2*nobs+5]<0,2*nobs+5]=0                                                 # filter negative gain
dat[:,2*nobs+6]=(1-np.sqrt(np.diag(A))/np.sqrt(np.diag(B)))*100                   # error reduction
dat[:,2*nobs+7]=dat[:,2*nobs+3]-dat[:,0]                                          # posterior noise

Chi_sq = (obs[3,:]-obs[1,:]).dot(tem_4).dot(np.transpose(obs[3,:]-obs[1,:]))/nobs # Chi-square
DFS = np.trace(K.dot(H))*100/nobs                                                 # Degree of freedom

#=====================posterior DXCO2=======================================   
for j in range(nobs):
    obs[4,j]=sum(dat[:,2*nobs+3]*dat[:,j+1])              # posterior DXCO2

#=====================save results and statistical analyses=================
res=np.full((num_p,8), np.nan) 
sts=np.full((3,5), np.nan) 

res[inx_2,0]=dat[:,0]                     # truth
res[inx_2,1]=dat[:,2*nobs+2]              # prior flux
res[inx_2,2]=dat[:,2*nobs+3]              # posterior flux
res[inx_2,3]=dat[:,2*nobs+4]              # flux correction
res[inx_2,4]=dat[:,2*nobs+5]              # gain
res[inx_2,5]=dat[:,2*nobs+6]              # error reduction
res[inx_2,6]=dat[:,2*nobs+1]              # prior flux noise
res[inx_2,7]=dat[:,2*nobs+7]              # posterior flux noise

sts[0,0] = round(np.nansum(res[:,0])*44*(1e-6),2)  # total true emissions, unit conversion (tonneCO2 s-1)
sts[1,0] = round(np.nansum(res[:,1])*44*(1e-6),2)  # total prior emissions
sts[2,0] = round(np.nansum(res[:,2])*44*(1e-6),2)  # total posterior emissions

sts[0,1] = round(np.nanmean(res[:,5]),2)           # mean error reduction
sts[1,1] = round(flux_rerr*nflux*44*(1e-6),2)      # prior random flux error
sts[2,1] = round(np.sqrt((np.mean(np.sqrt(np.diag(A)))*nflux*44*(1e-6))**(2)-(sts[2,0]-sts[0,0])**(2)),2)  #posterior random flux error

sts[0,2] = round(Chi_sq,2)                         # Chi-square
sts[1,2] = round(DFS,2)                            # degree of freedom signal
sts[2,2] = round(nflux,2)                          # number of flux pixel

sts[0,3] = obs_bias                       # obs bias
sts[1,3] = obs_rerr                       # obs random error
sts[2,3] = nobs                           # number of obs

sts[0,4] = flux_bias                      # flux bias
sts[1,4] = flux_rerr                      # flux random error
sts[2,4] = correl_flag                    # spatial correlation length

np.save('/home/kwu2/res/2021/inversion/0519/dat/output/flux_4.npy',res)
np.save('/home/kwu2/res/2021/inversion/0519/dat/output/obs_4.npy',obs)
np.save('/home/kwu2/res/2021/inversion/0519/dat/output/loc_4.npy',loc)
np.save('/home/kwu2/res/2021/inversion/0519/dat/output/sts_4.npy',sts)
