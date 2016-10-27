import matplotlib.pyplot as plt
from scipy.constants import *
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
from   astropy.io import fits
from   numpy import random
import scipy.optimize
import astropy.units as u
import sys
from   astropy.io import ascii
from astropy.convolution import Gaussian1DKernel, convolve


# ----- read file --------

filename = '1342238709_HR_spectrum_point.fits' 
spec     = fits.open(filename)

centreDetectors = ["SLWC3","SSWD4"]

## -- define the SLW and SSW bands in the central pixel 

cent_spec_SLW   = np.full((200,3,1905),1.0) 
cent_spec_SSW   = np.full((200,3,2082),1.0) 



## -- define the background (baseline) of the SLW and SSW bands of the central pixel 

back_spec_SLW   = np.full((200,6,3,1905),1.0) 
back_spec_SSW   = np.full((200,9,3,2082),1.0) 

# -- detector name lists -- 
offAxisDets = {'SSW':['SSWB2', 'SSWB3', 'SSWB4', 'SSWC2', 'SSWC5', 'SSWD2', 'SSWD6', 'SSWE5', 'SSWF3'], 'SLW':['SLWD3', 'SLWC4', 'SLWB3', 'SLWB2', 'SLWC2', 'SLWD2']}


slw_list = offAxisDets['SLW']
ssw_list = offAxisDets['SSW']


#  --- 200 subscans per scan (fits file)
#  --- every 25 headers, loop 24 pixels  

for i in range(0, 200):   # loop 200 subscans 
    idx= i*25+2
    a=spec[idx:idx+24]
    n = 0
    m = 0 
    for k in range(0, 24):  # loop 24 pixels 
        if (a[k].header['EXTNAME'] == centreDetectors[0]):
            cent_spec_SLW[i,0,]      = a[k].data.wave 
            cent_spec_SLW[i,1,]      = a[k].data.flux 
            cent_spec_SLW[i,2,]      = a[k].data.error 

        if (a[k].header['EXTNAME'] in slw_list): 
            back_spec_SLW[i,n,0, ]   = a[k].data.wave 
            back_spec_SLW[i,n,1, ]   = a[k].data.flux 
            back_spec_SLW[i,n,2, ]   = a[k].data.error 
            n = n +1

        if (a[k].header['EXTNAME'] == centreDetectors[1]):
            cent_spec_SSW[i,0,]      = a[k].data.wave 
            cent_spec_SSW[i,1,]      = a[k].data.flux 
            cent_spec_SSW[i,2,]      = a[k].data.error 

        if (a[k].header['EXTNAME'] in ssw_list): 
            back_spec_SSW[i,m,0, ]   = a[k].data.wave 
            back_spec_SSW[i,m,1, ]   = a[k].data.flux 
            back_spec_SSW[i,m,2, ]   = a[k].data.error 
            m = m +1


# ---  define mean/median value arrays with a dimension of (3,1905)  
SLW_mean        = np.full((3,1905),1.0) 
SLW_median      = np.full((3,1905),1.0) 
back_SLW_mean   = np.full((3,1905),1.0) 



# ---  for each channel (1905), calculate the mean/median  values 
for i in range(0, 1905):
      SLW_mean[0,i]      =   np.mean(cent_spec_SLW[:,0,i])    # -- mean   x-axis (freq) 
      SLW_mean[1,i]      =   np.mean(cent_spec_SLW[:,1,i])    # -- mean   y-axis (flux) 
      SLW_mean[2,i]      =    np.std(cent_spec_SLW[:,1,i])    # -- mean   z-axis (error in flux)
      SLW_median[0,i]    = np.median(cent_spec_SLW[:,0,i])    # -- median x-axis (freq)                 
      SLW_median[1,i]    = np.median(cent_spec_SLW[:,1,i])    # -- median y-axis (flux) 
      SLW_median[2,i]    =    np.std(cent_spec_SLW[:,1,i])    # -- median z-axis (error in flux)
      back_SLW_mean[0,i] = np.mean(back_spec_SLW[:,:,0,i])    # -- mean background x-axis (freq)                  
      back_SLW_mean[1,i] = np.mean(back_spec_SLW[:,:,1,i])    # -- mean background y-axis (flux) 
      back_SLW_mean[2,i] =  np.std(back_spec_SLW[:,:,1,i])    # -- mean background z-axis (error in flux)



# --- mean value of 200 SLW scans without subtraction 
plt.clf()
fig, ax_f = plt.subplots()
ax_f.plot(SLW_mean[0], SLW_mean[1], label='mean')
ax_f.set_ylim(-1, 1)
plt.legend( loc=2, borderaxespad=0.)
plt.savefig('SLW_mean.pdf')
#----------------------------------------------------


# --- median value of 200 SLW scans without subtraction 
plt.clf()
fig, ax_f = plt.subplots()
ax_f.plot(SLW_median[0], SLW_median[1], label='median')
ax_f.set_ylim(-1, 1)
plt.legend( loc=2, borderaxespad=0.)
plt.savefig('SLW_median.pdf')
#----------------------------------------------------

# --- std value of 200 SLW scans without subtraction 
plt.clf()
fig, ax_f = plt.subplots()
ax_f.plot(SLW_mean[0], SLW_mean[2], label='std')
ax_f.set_ylim(-1, 8)
plt.legend( loc=2, borderaxespad=0.)
plt.savefig('SLW_std.pdf')
#----------------------------------------------------


# --- std value of 200 SLW scans without subtraction 
plt.clf()
fig, ax_f = plt.subplots()
ax_f.plot(back_SLW_mean[0], back_SLW_mean[1], label='back-SLW')
ax_f.set_ylim(-1, 1)
plt.legend( loc=2, borderaxespad=0.)
plt.savefig('back_SLW.pdf')
#----------------------------------------------------

# --define a Gaussian with std=21 GHz, convolve it with the spectra, to define the local baselines as the background.
g       = Gaussian1DKernel(stddev=21)
back_sm = convolve(back_SLW_mean[1],g, boundary='extend')
plt.clf()
fig, ax_f = plt.subplots()
ax_f.plot(back_SLW_mean[0], back_sm, label='back-SLW-sm')
ax_f.set_ylim(-1, 1)
plt.legend( loc=2, borderaxespad=0.)
plt.savefig('back_SLW_sm.pdf')


# -- b is the local baseline background subtracted spectrum, using the mean of all 200 subscan spectra   
b = SLW_mean[1]-back_sm
plt.clf()
fig, ax_f = plt.subplots()
ax_f.plot(SLW_mean[0],b , label='mean-sub')
ax_f.set_ylim(-1,1) 
plt.legend( loc=2, borderaxespad=0.)
plt.savefig('SLW_mean_bg_sub.pdf')


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

#  --- define baseline subtracted spectral arrays  
bs_SLW      = np.full((200,3,1905),1.0) 
bs_SLW_mean = np.full((3,1905),1.0) 


for i in range(0, 200):
    idx= i*25+2
    a=spec[idx:idx+24]
    n = 0
    m = 0 
    for k in range(0, 24):
        if (a[k].header['EXTNAME'] == centreDetectors[0]):
            cent_spec_SLW[i,0,]      = a[k].data.wave 
            cent_spec_SLW[i,1,]      = a[k].data.flux 
            cent_spec_SLW[i,2,]      = a[k].data.error 
            back_sm = convolve(cent_spec_SLW[i,1,], g, boundary='extend')
            bs_SLW[i,0:]             = a[k].data.wave
            bs_SLW[i,1:]             = cent_spec_SLW[i,1,]-back_sm 


    plt.clf()
    fig, ax_f = plt.subplots()
    ax_f.plot(cent_spec_SLW[i,0,],cent_spec_SLW[i,1,]-back_sm)
    ax_f.set_ylim(-5, 5)
    plt.savefig('/subscan_plots/SLW_plot_bs_'+str(i)+'.pdf')
 #----------------output each scan plot 
    plt.clf()
    fig, ax_f = plt.subplots()
    ax_f.plot(cent_spec_SLW[i,0,],cent_spec_SLW[i,1,])
    ax_f.plot(cent_spec_SLW[i,0,],back_sm)
    ax_f.set_ylim(-5, 5)
    plt.savefig('/subscan_plots/SLW_plot_bs_bl_'+str(i)+'.pdf')



for i in range(0, 1905):
      bs_SLW_mean[0,i] = np.mean(bs_SLW[:,0,i])
      bs_SLW_mean[1,i] = np.mean(bs_SLW[:,1,i])
      bs_SLW_mean[2,i] =  np.std(bs_SLW[:,1,i])


plt.clf()
fig, ax_f = plt.subplots()
ax_f.plot(bs_SLW_mean[0], bs_SLW_mean[1], label='bl_scan_mean')
ax_f.set_ylim(-1,1) 
plt.legend( loc=2, borderaxespad=0.)
plt.savefig('SLW_bs_mean.pdf')

plt.clf()
fig, ax_f = plt.subplots()
ax_f.plot(bs_SLW_mean[0], bs_SLW_mean[2], label='std')
ax_f.set_ylim(-1,6) 
plt.legend( loc=2, borderaxespad=0.)
plt.savefig('SLW_bs_std.pdf')


plt.clf()
plt.hist(cent_spec_SLW[:,1,1000], bins=np.linspace(-2,2,20),color='green',alpha=0.5)
plt.hist(       bs_SLW[:,1,1000], bins=np.linspace(-2,2,20),color='red',alpha=0.5)
plt.savefig('error_histogram.pdf')

plt.clf()
p0 = 2
p1 = 800
p2 = 0.37733
x  = bs_SLW[:,0,1000]
fig, ax_f = plt.subplots()
sinconly   = p0*np.sinc((SLW_mean[0] -p1)/p2)
spec_n_sinc= p0*np.sinc((SLW_mean[0] -p1)/p2)+SLW_mean[1]
ax_f.plot(SLW_mean[0], sinconly)
ax_f.plot(SLW_mean[0], spec_n_sinc)
plt.savefig('sinc.pdf')

spec_only  = cent_spec_SLW[100,1,] #SLW_mean[1] 
sinc_only  = p0*np.sinc((SLW_mean[0] -p1)/p2)
spec_n_sinc= p0*np.sinc((SLW_mean[0] -p1)/p2)+spec_only
spec_sm           = convolve(spec_only  , g, boundary='extend')
spec_n_sinc_sm    = convolve(spec_n_sinc, g, boundary='extend')
spec_n_sinc_sm_bs = spec_n_sinc - spec_n_sinc_sm  


plt.clf()
fig, ax_f = plt.subplots()
ax_f.plot(bs_SLW_mean[0], spec_n_sinc        , label='spec')
ax_f.plot(bs_SLW_mean[0], spec_sm            , label='spec_sm')
ax_f.plot(bs_SLW_mean[0], sinc_only          , label='sinc')
ax_f.plot(bs_SLW_mean[0], spec_n_sinc_sm     , label='spec_n_sinc_sm')
ax_f.plot(bs_SLW_mean[0], spec_n_sinc_sm_bs  , label='spec_n_sinc_sm_bs')
ax_f.set_ylim(-3,3) 
ax_f.set_xlim(750,850) 
plt.legend( loc=2, borderaxespad=0.)
plt.savefig('test_sm_bs.pdf')





#   cent_spec_SLW.append([SLW_wave,SLW_flux]) 
#   cent_spec_SSW.append([SSW_wave, SSW_flux]) 
##  -- define a plot
#    fig, ax_f = plt.subplots()
#    ax_f.plot(SLW_wave, SLW_flux)
#    ax_f.set_ylim(-5, 5)
#    plt.savefig('SLW_plot'+str(i)+'.pdf')
#    plt.clf()
#    fig, ax_f = plt.subplots()
#    ax_f.plot(SSW_wave, SSW_flux)
#    ax_f.set_ylim(-5, 5)
#    plt.savefig('SSW_plot'+str(i)+'.pdf')
 


