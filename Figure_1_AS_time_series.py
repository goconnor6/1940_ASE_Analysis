#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:10:02 2020

Plota 3-panel figure 1:
Timeseries of SLP and U averaged over ASE shelf break region from from 1900 to 2005
with 2 recons and ERA5. Shifts anom ref period in recons to 1979-2005. 
Includes corr, sig, and CE in legend. 

Also calculates the magnitude of the 1940 event, classified in 10 different ways
And whether the 1940 event is unique in the 20th century 
(and whether the trend influences its uniqueness)

@author: gemma
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from Functions_1940_analysis import load_1d_data, calc_1d_corr, calc_1d_ce
from Datasets_1940_Analysis import *



#%% load data 

anom_ref = 1979,2005
region = 'ASE SB'


recon_start,recon_stop = 1900,2005
time_per = recon_start,recon_stop

#CESM recons
cesm_psl_path = cesm_recon.path.replace('*vname*','psl')
cesm_u10_path = cesm_recon.path.replace('*vname*','u10')
cesm_g_u10_path = cesm_recon.path.replace('*vname*','g_u10')
cesm_coral_psl_path = cesm_coral_recon.path.replace('*vname*','psl')

cesm_recon_psl = load_1d_data( cesm_psl_path, 'psl', region, time_per = time_per, \
                                  anom_ref = anom_ref, np_array = True)
cesm_recon_u = load_1d_data( cesm_u10_path, 'u10', region, time_per = time_per, \
                                  anom_ref = anom_ref, np_array = True)
cesm_recon_g_u = load_1d_data( cesm_g_u10_path, 'g_u10', region, time_per = time_per, \
                                  anom_ref = anom_ref, np_array = True)
cesm_coral_recon_psl = load_1d_data( cesm_coral_psl_path, 'psl', region,\
                                    time_per = time_per, anom_ref = anom_ref, np_array = True)
    
#PACE recons
pace_psl_path = pace_recon.path.replace('*vname*','psl')
pace_u10_path = pace_recon.path.replace('*vname*','u10')
pace_g_u10_path = pace_recon.path.replace('*vname*','g_u10')

pace_recon_psl = load_1d_data( pace_psl_path, 'psl', region, time_per = time_per, \
                                   anom_ref = anom_ref, np_array = True)
pace_recon_u = load_1d_data( pace_u10_path, 'u10', region, time_per = time_per, \
                                   anom_ref = anom_ref, np_array = True)
pace_recon_g_u = load_1d_data( pace_g_u10_path, 'g_u10', region, time_per = time_per, \
                                   anom_ref = anom_ref, np_array = True)

recon_time = np.linspace(recon_start,recon_stop,recon_stop - recon_start + 1)


#get ERA5 
era_psl = load_1d_data( verif_dir + '/ERA5/annual_psl_1979_2019.nc', 'psl', \
                                region, [1979,2005], anom_ref = anom_ref, np_array = True)
era_u = load_1d_data( verif_dir + '/ERA5/annual_u1000_1979_2019.nc', 'u1000', \
                               region, [1979,2005],anom_ref = anom_ref, np_array = True)
era_time = np.linspace(1979, 2005, 2005-1979+1)

#Calc nino3.4 index
# nino_time,nino34_idx = calc_nino_idx('Nino3.4','v5',time_per)
nino34_data = load_1d_data(verif_dir + 'annual_ersstv5_1854_2019.nc','sst',\
                          'Nino3.4',time_per)
nino34_idx = nino34_data - nino34_data.mean()
    

#%% Calc stats with ERA5 over period of overlap


era_stop_idx = 2005 - 1979 + 1 #+1 so you include the stat_stop year
recon_i1 = 1979 - recon_start
recon_i2 = 2005 - recon_start + 1 #+1 to include stat_stop year


data_list = [cesm_recon_psl, pace_recon_psl,\
            cesm_recon_u, pace_recon_u,\
            cesm_recon_g_u, pace_recon_g_u]
era_list = [era_psl,era_psl,era_u,era_u,era_u,era_u]
lab_list = [cesm_recon.name,pace_recon.name]*3
recon_stats = []

for dataset,era_data,lab in zip(data_list,era_list,lab_list):
    
    corr,sig = calc_1d_corr(dataset[recon_i1:recon_i2],
                            era_data[0:era_stop_idx])
    ce = calc_1d_ce(dataset[recon_i1:recon_i2],\
                    era_data[0:era_stop_idx])
    stat_str = lab + f'\nr={corr}{sig}, CE={ce}'
    recon_stats.append(stat_str)


#%% Make Figure

fs = 9 
lw = 1.5

fig = plt.figure()
fig.set_size_inches(4,4.5)

#plot SLP----------------------------------
ax0 = fig.add_subplot(311)
ax0.plot(recon_time, cesm_recon_psl, color = cesm_recon.psl_color, \
         label = recon_stats[0], linewidth = lw,zorder=1)
ax0.plot(recon_time, pace_recon_psl, color = pace_recon.psl_color, \
         label = recon_stats[1], linewidth = lw,zorder = 0)
ax0.plot(era_time,era_psl,color = 'black',label = 'ERA5',linewidth = lw-.2)
ax0.set_ylim([-8.7,6.5])
ax0.set_yticks([-5,-2.5,0,2.5,5])
ax0.set_ylabel('SLP anomaly (hPa)')

#plot U10---------------------------------
ax1 = fig.add_subplot(312)
ax1.plot(recon_time, cesm_recon_u, color = cesm_recon.u10_color,zorder=1,\
         label = recon_stats[2], linewidth = lw)
ax1.plot(recon_time, cesm_recon_g_u, color = cesm_recon.g_u10_color,zorder=0,\
         label = recon_stats[4].split('\n')[-1], linewidth = lw-.3,linestyle = '-')
ax1.plot(recon_time, pace_recon_u, color = pace_recon.u10_color,zorder = 1,\
         label = recon_stats[3], linewidth = lw)
ax1.plot(recon_time, pace_recon_g_u, color = pace_recon.g_u10_color,zorder = 0,\
         label = recon_stats[5].split('\n')[-1], linewidth = lw-.3,linestyle = '-')
ax1.plot(era_time,era_u,color = 'black',label = 'ERA5',linewidth = lw-0.2)
ax1.set_ylim(-2.5,1.7)
ax1.set_yticks([-1.2,-0.6,0,0.6,1.2])
ax1.set_ylabel(r'U$_S$ anomaly (m/s)')

#plot Nino3.4 index and detrended sLP Recon------------------
ax2  =  fig.add_subplot(313)
#detrend SLP recon
cesm_recon_psl_dt = signal.detrend(cesm_recon_psl)
l1  =  ax2.plot(recon_time, cesm_recon_psl_dt, color = cesm_recon.psl_color,\
              label = cesm_recon.name)
recon_coral_psl_dt = signal.detrend(cesm_coral_recon_psl)
l3  =  ax2.plot(recon_time,recon_coral_psl_dt,\
                color = cesm_coral_recon.psl_color,label = cesm_recon.name+', corals only')
ax2.set_ylim([-8,5.8])
ax2.set_yticks([-4,-2,0,2,4])
ax2.set_ylabel('SLP anomaly (hPa)',fontsize = fs-1)
ax2.set_xlim([recon_start-2, recon_stop+2])
ax2.set_xlabel('Year',fontsize = fs-1,labelpad = 2)
ax2.tick_params(axis = 'x', labelsize = fs-2)
ax2.tick_params(axis = 'y', labelsize = fs-2)

ax3  =  ax2.twinx()
#detrend nino3.4 index
nino34_idx_dt  =  signal.detrend(nino34_idx)
#move anom ref period for nino index to 1979-2005
nino34_idx_anom  =  nino34_idx_dt - np.mean(nino34_idx_dt[79:])
l2  =  ax3.plot(recon_time,nino34_idx_anom,color = 'gray',\
              label = 'ERSSTv5 Nino3.4 Index',linewidth = lw-.2,zorder = 0)
ax3.set_ylim([-2.9,1.8])
ax3.set_yticks([-2,-1,0,1])
ax3.tick_params(pad=.1)
ax3.tick_params(axis = 'y', labelsize = fs-2)
ax3.set_ylabel('Nino3.4 Index',fontsize = fs-1,labelpad=1)


#format all subplots--------------------------
for ax in [ax0,ax1,ax2]:
    ax.grid(linestyle = '-', linewidth = '0.3',color = 'gray')  
    ax.set_xlim([recon_start-2,recon_stop+2])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()): 
        label.set_fontsize(fs-2)
    ax.set_ylabel(ax.get_ylabel(),fontsize=fs-1,labelpad=-.15)
    if ax == ax0 or ax == ax1:
        ax.legend(fontsize = fs-3,loc = 'lower left',ncol = 3,frameon = True,\
               columnspacing = 1,handletextpad  =  0.4)
    else:
        lns = l1 + l3 + l2
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, fontsize  =  fs-3,loc = 'lower left',ncol = 2,\
                   frameon = True,columnspacing = 1,handletextpad  =  0.4)
            
plt.subplots_adjust(wspace = 0.04,hspace = 0.22,top = 0.98,\
                    bottom = 0.08,right = 0.9,left = 0.12)


fig.text(0.125,0.95,'a)',fontsize=fs,weight='bold')
fig.text(0.125,0.625,'b)',fontsize=fs,weight='bold')
fig.text(0.125,0.31,'c)',fontsize=fs,weight='bold')

# plt.savefig('Plots/Figure_1_ASE_timeseries.png', bbox_inches = 'tight',dpi = 600)


#%% Determine whether 1940 event is unique in the 20th century


# Characterize the 1940 event in several different ways
# i.e. an event lasting 1 - n yrs long centered around 1940
def calc_nyr_means(n_events,data):
    """
    calculate means of all n-year-long events in reconstruction 
    We use full overlap because we want to sample all possibilities

    Parameters
    ----------
    n_events : int
    data : 1d np array 
        contains reconstruction data in ASE region starting in 1900

    Returns
    -------
    recon_1940_event_mags : list of floats, list length = n_events
        Contains magnitudes of the 1940 event for all definitions of the event
    start_date_larger_events : list of length n_lists. 
        each item in list is a list containing start dates of events exceeding 
        the 1940 magnitude 
        

    """
    #calculate magnitude of 1940 event in recon
    recon_1940_event_mags = []
    for n in range(1,n_events+1):
        if n%2 == 0:
            start_idx = int(40 - (n/2)) + 1
            n_event_data = data[ start_idx : start_idx + n ]
            # print(n, recon_time [start_idx : start_idx + n ])
        else:
            start_idx = 40 - (n//2) 
            n_event_data = data[start_idx : start_idx + n ]
            # print(n, recon_time [start_idx : start_idx + n ])
        event_mean = np.mean(n_event_data)
        recon_1940_event_mags.append(event_mean)
            
    #calculate magnitudes of all events in record
    #keep track of which events are bigger than recon
    start_date_larger_events = []
    for n in range(1,n_events+1):
        
        #check whether it's greater than the 1940 mean and where it is
        n_yr_large_event_starts = []
        n_yr_1940_mag = recon_1940_event_mags[n - 1]
        i = 0 
        while i < len(data) - n:
            
            mean_i = np.mean(data[i: i + n])
            if mean_i > n_yr_1940_mag:
                if recon_time[i] not in np.linspace(1934,1944,n_events+1):
                    #note if it's a separate event from the 1940 event
                    n_yr_large_event_starts.append(recon_time[i])

            i += 1
        start_date_larger_events.append(n_yr_large_event_starts)
    
    return recon_1940_event_mags, start_date_larger_events
    return recon_1940_event_mags, start_date_larger_events

n_events = 10
labs = ['CESM SLP recon','PACE SLP recon',\
        'CESM U recon','PACE U recon',\
        'CESM geo-U recon','PACE geo-U recon']
datasets = [cesm_recon_psl,pace_recon_psl,\
            cesm_recon_u,pace_recon_u,\
            cesm_recon_g_u,pace_recon_g_u]
    
for lab, data in zip(labs, datasets):
    
    print('\n'+lab+':')
    recon_mags_1940, recon_large_start_dates = calc_nyr_means(n_events, data)
    
    #repeat with detrended dataset to see whether it's dependent on trend
    dt_recon_mags_1940, dt_recon_large_start_dates = calc_nyr_means(n_events, \
                                                signal.detrend(data))
        
    # Show which types of 1940 event characterizations are unique
    for m in range(n_events):
        if len(recon_large_start_dates[m]) == 0:
            print(f'For {m+1}-year events, the 1940 event is unique')
            if len(dt_recon_large_start_dates[m]) != 0:
                print(f'But when you detrend, these {m}-year events are larger than 1940:',\
                      dt_recon_large_start_dates[m])
        else:
            print(f'These are the start dates of the {m}-yr events greater than 1940:',\
                  recon_large_start_dates[m])
    
            


