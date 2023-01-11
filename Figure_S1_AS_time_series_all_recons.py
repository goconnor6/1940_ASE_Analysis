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
from Functions_1940_analysis import load_1d_data, load_1d_data_raw_times, calc_1d_corr, calc_1d_ce
from Datasets_1940_Analysis import *



#%% load data 

# Initial Parameters
anom_ref = 1979,2000 # Stops in 2000 because Dalaiden 2021 recon stops then
region = 'ASE SB'
recon_start,recon_stop = 1900,2005 # For OConnor recon
time_per = recon_start,recon_stop

# get ERA5 
era_psl = load_1d_data( verif_dir + '/ERA5/annual_psl_1979_2019.nc', 'psl', \
                                region, [1979,2005], anom_ref = anom_ref, np_array = True)
era_u10 = load_1d_data( verif_dir + '/ERA5/annual_u1000_1979_2019.nc', 'u1000', \
                               region, [1979,2005],anom_ref = anom_ref, np_array = True)
era_time = np.linspace(1979, 2005, 2005-1979+1)


# Get O'Connor 2021 recon data and calc stats with era5
recons = [cesm_recon,hadcm_recon,pace_recon,lens_recon]
psl_data = []
u10_data = []
for recon in recons:

    recon_psl = load_1d_data(recon.path.replace('*vname*','psl'),\
                             'psl', region, time_per = time_per, \
                             anom_ref = anom_ref, np_array = True)
    recon_u10 = load_1d_data(recon.path.replace('*vname*','u10'),\
                             'u10', region, time_per = time_per, \
                             anom_ref = anom_ref, np_array = True)
    psl_data.append(recon_psl)
    u10_data.append(recon_u10)
o21_time = np.linspace(recon_start,recon_stop,recon_stop - recon_start + 1)
        
   # Get Dalaiden 2021 recon 
dal_time,dal_psl = load_1d_data_raw_times(dal_recon.path.replace('*vname*','SLP'),\
                                          'SLP',[1900,2000],region,anom_ref=anom_ref)
dal_time,dal_u10 = load_1d_data_raw_times(dal_recon.path.replace('*vname*','850hpa-Uwind'),\
                                          '850hpa-Uwind',[1900,2000],region,anom_ref=anom_ref)
psl_data.append(dal_psl)
u10_data.append(dal_u10)

#%% Stats with ERA5

labs = ["O'Connor 2021 recon\nCESM LM prior (natural)",\
        "O'Connor 2021 recon\nHadCM3 prior (natural)",\
        "O'Connor 2021 recon\nPACE prior (anthro)",\
        "O'Connor 2021 recon\nLENS prior (anthro)",\
        "Dalaiden 2021 recon\nCESM LM prior (natural)"] 
n_recons = len(labs)
psl_labs = []
u10_labs = []
# Get indices of datasets for calculating stats
era_stop_idx = 2000 - 1979 + 1 
recon_i1 = 1979 - recon_start
recon_i2 = 2000 - recon_start + 1

for i in range(n_recons):

    recon_psl = psl_data[i]
    psl_corr,psl_sig = calc_1d_corr(recon_psl[recon_i1:recon_i2],
                            era_psl[0:era_stop_idx])
    psl_ce = calc_1d_ce(recon_psl[recon_i1:recon_i2],\
                    era_psl[0:era_stop_idx])
    psl_stat_str = labs[i] + f'\nr={psl_corr}{psl_sig}, CE={psl_ce}'
    psl_labs.append(psl_stat_str)
    
    recon_u10 = u10_data[i]
    u10_corr,u10_sig = calc_1d_corr(recon_u10[recon_i1:recon_i2],
                            era_u10[0:era_stop_idx])
    u10_ce = calc_1d_ce(recon_u10[recon_i1:recon_i2],\
                    era_u10[0:era_stop_idx])
    u10_stat_str = labs[i] + f'\nr={u10_corr}{u10_sig}, CE={u10_ce}'
    u10_labs.append(u10_stat_str)


#%%plot reconstructions and ERA5

fs = 10 
lw = 1.5

fig = plt.figure()
fig.set_size_inches(6,5)
fig.add_subplot(211)

#plot SLP-----------------

colors = [cesm_recon.psl_color,'#2171b5',pace_recon.psl_color,'#8c96c6','#88419d']
z= [1,0,0,0,0]
for i in range(n_recons):
    try:
        plt.plot(o21_time,psl_data[i],color=colors[i],label=psl_labs[i],linewidth=lw,zorder=z[i])
    except:
        plt.plot(dal_time,psl_data[i],color=colors[i],label=psl_labs[i],linewidth=lw,zorder=z[i])
plt.plot(era_time,era_psl,color='black',label='ERA5',linewidth=lw)
plt.grid(linestyle='-', linewidth='0.3',color='gray')


plt.legend(fontsize = fs-3,loc='lower left',ncol=3,frameon = True,\
           columnspacing=1,handletextpad = 0.4)    
plt.xlim([recon_start-2,recon_stop+2])
plt.ylim([-13,8.1])
plt.xticks(fontsize=fs-2)
#plt.rcParams['xtick.major.pad']='0.1'
plt.yticks(fontsize=fs-2)
plt.ylabel('SLP anomaly (hPa)',fontsize=fs-1,labelpad=0)
plt.xlabel('Year',fontsize=fs-1,labelpad=0.5)

#plot U10----------------
fig.add_subplot(212)

colors = [cesm_recon.u10_color,'#fec44f',pace_recon.u10_color,'#78c679','#238443']
z = [1,0,0,0,0]
for i in range(n_recons):
    try:
        plt.plot(o21_time,u10_data[i],color=colors[i],label=u10_labs[i],linewidth=lw,zorder=z[i])
    except:
        plt.plot(dal_time,u10_data[i],color=colors[i],label=u10_labs[i],linewidth=lw,zorder=z[i])
plt.plot(era_time,era_u10,color='black',label='ERA5',linewidth=lw)
plt.grid(linestyle='-', linewidth='0.3',color='gray')
plt.ylim([-3.2,1.9])
plt.grid(linestyle='-', linewidth='0.3',color='gray')

plt.legend(fontsize = fs-3,loc='lower left',ncol=3,frameon = True,\
           columnspacing=1,handletextpad = 0.4)    
plt.xlim([recon_start-2,recon_stop+2])
plt.xticks(fontsize=fs-2)
#plt.rcParams['xtick.major.pad']='0.1'
plt.yticks(fontsize=fs-2)
plt.ylabel(r'U$_S$ anomaly (m/s)',fontsize=fs-1,labelpad=0)
plt.xlabel('Year',fontsize=fs-1,labelpad=0.5)


plt.subplots_adjust(wspace=0.04,hspace=0.2,top = 0.98,bottom=0.1,right=0.98,left=0.09)

fig.text(0.095,0.945,'a',fontsize=fs,weight='bold')
fig.text(0.095,0.465,'b',fontsize=fs,weight='bold')

# plt.savefig('Plots/Figure_A1_ASE_timeseries.png', bbox_inches = 'tight',dpi = 600)

