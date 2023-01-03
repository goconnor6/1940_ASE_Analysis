#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:10:02 2020

Plot 2-panel figure:
timeseries of SLP and U averaged over PITT region from from 1900 to 2005
with 4 runs and ERA5. Shifts anom ref period in recons to 1979-2005. 
Includes corr, sig, and CE in legend. 
Option to calculate trends. 

@author: gemma
"""


import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats, signal
from Functions_1940_analysis import load_1d_recon, load_1d_era5, \
    load_1d_dal_recon, calc_1d_corr, calc_1d_ce



recon_time_per = 1900,2005
dal_time_per = 1900,2000
stat_time_per = 1979,2000 #stop in 2000 for comparisons to Dalaiden
anom_ref = 1979,2000
save_fig = False

dal = True
pace_ens_1 = False
pace_ens_6 = False
pace_ens_mean = False
era = True
calc_trends = False #last section option to print trend for each dataset


# load data and calc stats with ERA5---------------------------------------

#get ERA5 in shelf break region
era_time,era_psl = load_1d_era5('psl',[1979,2005],'ASE',anom_ref=anom_ref)
era_time,era_u = load_1d_era5('u1000',[1979,2005],'ASE',anom_ref=anom_ref)

#get indexes set up for stats calcs so selecting stat period data only
recon_start,recon_stop = recon_time_per
stat_start,stat_stop = stat_time_per
era_stop_idx = stat_stop - 1979 + 1 #+1 so you include the stat_stop year
recon_start_idx = 1979 - recon_start
recon_stop_idx = stat_stop - recon_start + 1 #+1 to include stat_stop year

#set up lists for populating data and stats strings
runs = [cesm_run, hadcm_run, lens_run, pace_run]
psl_data = []
u_data = []
psl_stat_str = []
u_stat_str = []

#get data and calc stats for each run
for run in runs:
    
    print(run)
    
    #SLP
    recon_time,recon_psl = load_1d_recon(run,'psl',recon_time_per,'ASE',anom_ref=anom_ref)
    psl_data.append(recon_psl)
    corr, sig = calc_1d_corr(recon_psl[recon_start_idx:recon_stop_idx],era_psl[0:era_stop_idx])
    ce = calc_1d_ce(recon_psl[recon_start_idx:recon_stop_idx],era_psl[0:era_stop_idx])
    stat_str = '\nr= '+str(corr)+sig+', CE= '+ce
    psl_stat_str.append(stat_str)
    
    #U
    recon_time,recon_u = load_1d_recon(run,'u10',recon_time_per,'ASE',anom_ref=anom_ref)
    u_data.append(recon_u)
    corr, sig = calc_1d_corr(recon_u[recon_start_idx:recon_stop_idx],era_u[0:era_stop_idx])
    ce = calc_1d_ce(recon_u[recon_start_idx:recon_stop_idx],era_u[0:era_stop_idx])
    stat_str = '\nr= '+str(corr)+sig+', CE= '+ce
    u_stat_str.append(stat_str)
    

if dal:
    #get Dalaiden 2021 SLP and calc stats
    dal_time,dal_psl = load_1d_dal_recon('SLP',dal_time_per,'ASE',anom_ref=anom_ref)
    corr, sig = calc_1d_corr(dal_psl[recon_start_idx:],era_psl[0:22])
    ce = calc_1d_ce(dal_psl[recon_start_idx:],era_psl[0:22])
    dal_psl_str = '\nr= '+str(corr)+sig+', CE= '+ce
    
    #get Dalaiden U and calc stats
    dal_time,dal_u = load_1d_dal_recon('850hpa-Uwind',dal_time_per,'ASE',anom_ref=anom_ref)
    corr, sig = calc_1d_corr(dal_u[recon_start_idx:],era_u[0:22])
    ce = calc_1d_ce(dal_u[recon_start_idx:],era_u[0:22])
    dal_u_str = '\nr= '+str(corr)+sig+', CE= '+ce


#get PACE ensemble 1
if pace_ens_1:
    #get SLP
    psl_path = 'Model/PAC_PACE/annual_psl_PAC_PACE_ens_01_1920_2005.nc'
    ds_psl = xr.open_dataset(psl_path)
    ds_psl = ds_psl.sel(time=slice(1920,stop),lat=slice(lat1,lat2),lon=slice(lon1,lon2))
    pace_time = ds_psl.time
    pace_psl = ds_psl.psl
    pace_psl = pace_psl/100
    pace_psl = np.mean(pace_psl,axis=1)
    pace_psl = np.mean(pace_psl,axis=1)
    mean_61_90 = np.mean(pace_psl[1961-start:1990-start+1])
    pace1_psl_anom = pace_psl - mean_61_90
    
    
    # #get U10
    u_path = 'Model/PAC_PACE/annual_u10_PAC_PACE_ens_01_1920_2005.nc'
    ds_u = xr.open_dataset(u_path)
    ds_u = ds_u.sel(time=slice(1920,stop),lat=slice(lat1,lat2),lon=slice(lon1,lon2))
    pace_u = ds_u.u10
    pace_u = np.mean(pace_u,axis=1)
    pace_u = np.mean(pace_u,axis=1)
    mean_61_90 = np.mean(pace_u[1961-start:1990-start+1])
    pace1_u_anom = pace_u - mean_61_90

#get PACE ensemble 6
if pace_ens_6:
    #get SLP
    psl_path = 'Model/PAC_PACE/annual_psl_PAC_PACE_ens_06_1920_2005.nc'
    ds_psl = xr.open_dataset(psl_path)
    ds_psl = ds_psl.sel(time=slice(1920,stop),lat=slice(lat1,lat2),lon=slice(lon1,lon2))
    pace_time = ds_psl.time
    pace_psl = ds_psl.psl
    pace_psl = pace_psl/100
    pace_psl = np.mean(pace_psl,axis=1)
    pace_psl = np.mean(pace_psl,axis=1)
    mean_61_90 = np.mean(pace_psl[1961-start:1990-start+1])
    pace6_psl_anom = pace_psl - mean_61_90
    
    
    # #get U10
    u_path = 'Model/PAC_PACE/annual_u10_PAC_PACE_ens_06_1920_2005.nc'
    ds_u = xr.open_dataset(u_path)
    ds_u = ds_u.sel(time=slice(1920,stop),lat=slice(lat1,lat2),lon=slice(lon1,lon2))
    pace_u = ds_u.u10
    pace_u = np.mean(pace_u,axis=1)
    pace_u = np.mean(pace_u,axis=1)
    mean_61_90 = np.mean(pace_u[1961-start:1990-start+1])
    pace6_u_anom = pace_u - mean_61_90
    


#plot reconstructions and ERA5---------------------------------------------------

fs = 10 
lw = 1.5

fig = plt.figure()
fig.set_size_inches(7.5,4)
fig.add_subplot(211)

#plot SLP-----------------

plt.plot(recon_time,psl_data[0],color='tab:blue',label='CESM LM recon'+psl_stat_str[0],linewidth=lw)
plt.plot(recon_time,psl_data[1],color='tab:orange',label='HadCM3 LM recon'+psl_stat_str[1],linewidth=lw)
plt.plot(recon_time,psl_data[2],color='tab:green',label='LENS recon'+psl_stat_str[2],linewidth=lw)
plt.plot(recon_time,psl_data[3],color='maroon',label='PACE recon'+psl_stat_str[3],linewidth=lw)
if dal:
    plt.plot(dal_time,dal_psl,color='gray',label='Dalaiden recon'+dal_psl_str,linewidth=lw)
if pace_ens_1:
    plt.plot(pace_time,pace1_psl_anom,color='navy',label='PACE ens 1',linewidth=lw)
if pace_ens_6:
    plt.plot(pace_time,pace6_psl_anom,color='navy',label='PACE ens 6',linewidth=lw)
if pace_ens_mean:
    plt.plot(pace_time,pace_em_psl_anom,color='navy',label='PACE ens mean',linewidth=lw)

# plt.plot(time,ghg_psl_anom,color='black',label='LENS GHG response',linewidth=lw)
if era:
    #era from [0:27] if just want till 2005
    plt.plot(era_time,era_psl,color='black',label='ERA5',linewidth=lw)
plt.grid(linestyle='-', linewidth='0.3',color='gray')


plt.legend(fontsize = fs-3,loc='lower left',ncol=7,frameon = True,\
           columnspacing=1,handletextpad = 0.4)    
plt.xlim([recon_start-2,recon_stop+2])
plt.ylim([-9.8,8.1])
plt.xticks(fontsize=fs-2)
#plt.rcParams['xtick.major.pad']='0.1'
plt.yticks(fontsize=fs-2)
plt.ylabel('SLP anomaly (hPa)',fontsize=fs-1,labelpad=0)
plt.xlabel('Year',fontsize=fs-1,labelpad=0.5)

#plot U10----------------
fig.add_subplot(212)

plt.plot(recon_time,u_data[0],color='tab:blue',label='CESM LM recon'+u_stat_str[0],linewidth=lw)
plt.plot(recon_time,u_data[1],color='tab:orange',label='HadCM3 LM recon'+u_stat_str[1],linewidth=lw)
plt.plot(recon_time,u_data[2],color='tab:green',label='LENS recon'+u_stat_str[2],linewidth=lw)
plt.plot(recon_time,u_data[3],color='maroon',label='PACE recon'+u_stat_str[3],linewidth=lw)
if dal:
    plt.plot(dal_time,dal_u,color='gray',label='Dalaiden recon'+dal_u_str,linewidth=lw)
if pace_ens_1:
    plt.plot(pace_time,pace1_u_anom,color='navy',label='PACE ens 1',linewidth=lw)
if pace_ens_6:
    plt.plot(pace_time,pace6_u_anom,color='navy',label='PACE ens 6',linewidth=lw)
if pace_ens_mean:
    plt.plot(pace_time,pace_em_u_anom,color='navy',label='PACE ens mean',linewidth=lw)

# plt.ylim([-2.3,1.4])
plt.ylim([-2.3,1.9])
if era:
    #[0:27] for end at 2005
    plt.plot(era_time,era_u,color='black',label='ERA5',linewidth=lw)
plt.grid(linestyle='-', linewidth='0.3',color='gray')

plt.legend(fontsize = fs-3,loc='lower left',ncol=7,frameon = True,\
           columnspacing=1,handletextpad = 0.4)    
plt.xlim([recon_start-2,recon_stop+2])
plt.xticks(fontsize=fs-2)
#plt.rcParams['xtick.major.pad']='0.1'
plt.yticks(fontsize=fs-2)
plt.ylabel(r'U$_S$ anomaly (m/s)',fontsize=fs-1,labelpad=0)
plt.xlabel('Year',fontsize=fs-1,labelpad=0.5)


plt.subplots_adjust(wspace=0.04,hspace=0.25,top = 0.98,bottom=0.1,right=0.99,left=0.06)

fig.text(0.075,0.935,'a',fontsize=fs)
fig.text(0.075,0.43,'b',fontsize=fs)

if save_fig:
    plt.savefig('Plots/Figure_S1_ASE_timeseries.png', bbox_inches = 'tight',dpi = 600)
        
#-----------------------------------------------stats
#calculate ensemble mean trend
if calc_trends:
    z = stats.linregress(time,psl_data[0])
    print('CESM recon SLP trend = '+"{:.2f}".format(z[0]*100)+ 'hPa/cent')
    z = stats.linregress(time,psl_data[1])
    print('HadCM3 recon SLP trend = '+"{:.2f}".format(z[0]*100)+ 'hPa/cent')
    z = stats.linregress(time,psl_data[2])
    print('LENS PI recon SLP trend = '+"{:.2f}".format(z[0]*100)+ 'hPa/cent')
    z = stats.linregress(time,psl_data[3])
    print('LENS recon SLP trend = '+"{:.2f}".format(z[0]*100)+ 'hPa/cent')
    z = stats.linregress(time,psl_data[4])
    print('PACE recon SLP trend = '+"{:.2f}".format(z[0]*100)+ 'hPa/cent')
    z = stats.linregress(dal_time,dal_psl_anom)
    print('Dalaiden recon SLP trend = '+"{:.2f}".format(z[0]*100)+ 'hPa/cent')
    
    z = stats.linregress(time,u_data[0])
    print('CESM recon US trend = '+"{:.2f}".format(z[0]*100)+ 'm/s/cent')
    z = stats.linregress(time,u_data[1])
    print('HadCM3 recon US trend = '+"{:.2f}".format(z[0]*100)+ 'm/s/cent')
    z = stats.linregress(time,u_data[2])
    print('LENS PI US trend = '+"{:.2f}".format(z[0]*100)+ 'm/s/cent')
    z = stats.linregress(time,u_data[3])
    print('LENS recon US trend = '+"{:.2f}".format(z[0]*100)+ 'm/s/cent')
    z = stats.linregress(time,u_data[4])
    print('PACE recon US trend = '+"{:.2f}".format(z[0]*100)+ 'm/s/cent')
    z = stats.linregress(dal_time,dal_u_anom)
    print('Dalaiden recon US trend = '+"{:.2f}".format(z[0]*100)+ 'm/s/cent')
    if pace_ens_1:
        z = stats.linregress(pace_time,pace1_psl_anom)
        print('PACE ens 1 SLP trend = '+"{:.2f}".format(z[0]*100)+ 'hPa/cent')
        z = stats.linregress(pace_time,pace1_u_anom)
        print('PACE ens 1 US trend = '+"{:.2f}".format(z[0]*100)+ 'm/s/cent')
    if pace_ens_6:
        z = stats.linregress(pace_time,pace6_psl_anom)
        print('PACE ens 6 SLP trend = '+"{:.2f}".format(z[0]*100)+ 'hPa/cent')
        z = stats.linregress(pace_time,pace6_u_anom)
        print('PACE ens 6 US trend = '+"{:.2f}".format(z[0]*100)+ 'm/s/cent')
    if pace_ens_mean:
        z = stats.linregress(pace_time,pace_em_psl_anom)
        print('PACE EM SLP trend = '+"{:.2f}".format(z[0]*100)+ 'hPa/cent')
        z = stats.linregress(pace_time,pace_em_u_anom)
        print('PACE EM US trend = '+"{:.2f}".format(z[0]*100)+ 'm/s/cent')



