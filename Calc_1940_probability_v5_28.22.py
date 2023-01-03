#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:39:39 2022

Calculating 1940 Probabilty.
We treat all data as a set of ensemble members, 
each of which is put in anomaly space (relative to the ref period in each member), 
then we remove the LENS EM to isolate the internal component
(except for LENS PI which has no anthro component). 

Also creates Figures 5, 6, 7, and table 1. 
    
@author: gemma
"""

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import AutoMinorLocator
import scipy.stats as stats
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
from Functions_1940_analysis_v2 import load_1d_data
from Datasets_1940_Analysis import *


#%% Set initial parameters


#whether to use geostrophic winds for either recon
cesm_geo = False
pace_geo = True

# set to true if you want to rescramble the ensemble members. otherwise, load prescrmabled arrays
fresh_scramble = False
# Need to mannually uncomment save commands to overwrite excisting scrambled arrays

time_per = [1920,2005]
anom_ref = [1961,1990] #uses 41st to 70th vals in LENS PI
n_draws = 300
region = 'ASE SB'

#how many ways to characterize the event (lasting 1-m years long)
n_events = 15

save_fig = False
plt.close('all')


#%% 1. Either freshly rescramble the reconstruction ensembles or load prescrambled ensembles

#First, load original reconstruction ensemble
cesm_recon_psl_ens = load_1d_data(cesm_recon.ens_path.replace('*vname*','psl'),\
                                  'psl', region, time_per = time_per)
pace_recon_psl_ens = load_1d_data(pace_recon.ens_path.replace('*vname*','psl'),\
                                  'psl',region, time_per = time_per)
cesm_recon_u10_ens = load_1d_data(cesm_recon.ens_path.replace('*vname*','u10'),\
                                  'u10', region, time_per = time_per)
pace_recon_u10_ens = load_1d_data(pace_recon.ens_path.replace('*vname*','u10'),\
                                  'u10', region, time_per = time_per)
if cesm_geo:
    cesm_recon_u10_ens = cesm_recon.get_geo_wind_ens(cesm_recon_u10_ens, time_per, region)
if pace_geo:
    pace_recon_u10_ens = pace_recon.get_geo_wind_ens(pace_recon_u10_ens, time_per, region)     

if fresh_scramble:
     
    scram_cesm_recon_psl = cesm_recon.calc_scram_ens(cesm_recon_psl_full_ens,n_draws)
    scram_cesm_recon_u = cesm_recon.calc_scram_ens(cesm_recon_u10_full_ens,n_draws) 
    scram_pace_recon_psl = pace_recon.calc_scram_ens(pace_recon_psl_full_ens,n_draws)
    scram_pace_recon_u = pace_recon.calc_scram_ens(pace_recon_u10_full_ens,n_draws) 
    
    # Change to True to overwrite existing scrambled arrays
    save_fresh_scramble = False
    if save_fresh_scramble:
        np.save('Data/Reconstruction/Scrambled_reconstructions/cesm_run_psl.npy',scram_recon_psl)
        np.save('Data/Reconstruction/Scrambled_reconstructions/cesm_run_u10.npy',scram_recon_u)
        np.save('Data/Reconstruction/Scrambled_reconstructions/cesm_run_g_u10.npy',scram_recon_u)
        np.save('Data/Reconstruction/Scrambled_reconstructions/pace_run_psl.npy',scram_recon_psl)
        np.save('Data/Reconstruction/Scrambled_reconstructions/pace_run_u10.npy',scram_recon_u)
        np.save('Data/Reconstruction/Scrambled_reconstructions/pace_run_g_u10.npy',scram_recon_u)

# Otherwise load prescrambled arrays
else:

    # They each have shape (n_draws, n_years)
    scram_cesm_recon_psl = np.load('Data/Reconstruction/Scrambled_reconstructions/cesm_run_psl.npy')
    scram_cesm_recon_u10 = np.load('Data/Reconstruction/Scrambled_reconstructions/cesm_run_u10.npy')
    if cesm_geo:
        scram_cesm_recon_u10 = np.load('Data/Reconstruction/Scrambled_reconstructions/cesm_run_g_u10.npy')
    scram_pace_recon_psl = np.load('Data/Reconstruction/Scrambled_reconstructions/pace_run_psl.npy')
    if pace_geo:
        scram_pace_recon_u10 = np.load('Data/Reconstruction/Scrambled_reconstructions/pace_run_g_u10.npy')
    else:
        scram_pace_recon_u10 = np.load('Data/Reconstruction/Scrambled_reconstructions/pace_run_u10.npy')
    recon_time = cesm_recon_psl_ens.time
    
    
        

#%% 2. Put scrambled recon members in anomaly space and isolate the internal component


#load LENS hist EM in anomaly space for isolating internal component in recons
lens_hist_em_psl_anom = load_1d_data(lens_hist.em_path.replace('*vname*','psl'),\
                        'psl', region, time_per = time_per, anom_ref = anom_ref)
lens_hist_em_u10_anom = load_1d_data(lens_hist.em_path.replace('*vname*','u10'),\
                        'u10', region, time_per = time_per, anom_ref = anom_ref)

#get index for 1961 for putting in anom space
ref_start_i = np.where(recon_time == anom_ref[0])[0][0]
ref_stop_i = np.where(recon_time == anom_ref[1])[0][0] + 1

def get_scram_nat_recon_anom(scram_recon_data,lens_em_anom):
    """
    Put each scrambled recon ensemble member in anomaly space 
    so that they are treated the same way as the models. 
    Also remove the anthropogenic component (LENS historical ens mean)
    to isolate the internal component in the reconstructions.
    

    Parameters
    ----------
    scram_recon_data : 2d np array with shape (n_draws, n_yrs)
        scrambled reconstruction members (either psl or u10)
    lens_em_anom : 2d array (either np array or xr data array okay)
        contains LENS historical ensemble mean array, either psl or u10

    Returns
    -------
    scram_recon_anom_nat : 2d np array with shape (n_draws, n_yrs)
        Contains internal component of scrambled reconstruction ensemble in anomaly space 

    """
    
    #for each draw, put in anom space then remove LENS EM
    scram_recon_anom_nat = np.zeros(scram_recon_data.shape) #n_draws, n_yrs
    draw = 0
    for ens_data in scram_recon_data:
        
        #put ens data in anom space the same way you do the models
        ens_data_ref = np.mean(ens_data[ref_start_i:ref_stop_i]) 
        ens_data_anom = ens_data - ens_data_ref
        
        #remove LENS EM and populate array
        scram_recon_anom_nat[draw,:] = ens_data_anom - lens_em_anom
        
        draw += 1
        
    return scram_recon_anom_nat
    
scram_cesm_recon_psl_nat = get_scram_nat_recon_anom(scram_cesm_recon_psl,lens_hist_em_psl_anom)
scram_pace_recon_psl_nat = get_scram_nat_recon_anom(scram_pace_recon_psl,lens_hist_em_psl_anom)

scram_cesm_recon_u10_nat = get_scram_nat_recon_anom(scram_cesm_recon_u10,lens_hist_em_u10_anom)
scram_pace_recon_u10_nat = get_scram_nat_recon_anom(scram_pace_recon_u10,lens_hist_em_u10_anom)

#%% 3. Calculate 1940 event magnitudes in reconstructions 


# For each scrambled tseries, calculate the mean value maintained for m years
# centered around 1940

events = np.linspace(1,n_events,n_events,dtype=int)
start_yrs = []
stop_yrs = []
for m in events:
    
    #center around 1940/41 for even years
    if m%2 == 0:
        start_yr = 1940 - (m//2) + 1
        
    #centered around 1940 for odd years
    else:
        start_yr = 1940 - (m//2)
    stop_yr = 1940 + (m//2)
    start_yrs.append(start_yr)
    stop_yrs.append(stop_yr)
stop_yrs
start_yrs

def calc_thresholds(recon_data,start_yrs,stop_yrs):
    """
    Calculate mean and 95% CI of value maintained for each m-yr event in the 
    scrambled reconstructions. 

    Parameters
    ----------
    years : 1d numpy array
        years corresponding to recon_data
    recon_data : 2d np array
        scramble reconstruction members. shape n_ensembles, n_yrs. 
    start_yrs : list
        starting years for each m-yr event
        length = number of m-yrs events
    stop_yrs : list
        stopping years (inclusive) for each m-yr event

    Returns
    -------
    mean_thresholds : list
        list of mean values maintained in the mean of all 300 recon members
        for each m-yr event. order follows order of events corresponding to
        start_yrs (i.e. 1,2,3...,10-yr events)
    low_thresholds : list
        lower bound of 95% confidence interval threshold for all events
    upp_thresholds : list
        upper bound of 95% confidence interval threshold for all events

    """
    n_events = len(start_yrs)
    
    #for each event type
    mean_thresholds = [] #len n_events
    low_thresholds = []
    upp_thresholds = []
    for i in range(n_events):

        
        #get start index and stop index for this event, centered around 1940
        start_idx = np.where(years == start_yrs[i])[0][0]
        stop_idx = np.where(years == stop_yrs[i]+1)[0][0]
        
        #thresholds for an m-yr event based on the {n_draws} tseries 
        scram_thresholds = [] 
        
        #for each scrambled tseries (t goes through each tseries)
        for tseries in recon_data:
            
            #get values during event
            tseries_event = tseries[start_idx:stop_idx]
            event_mean = np.mean(tseries_event)
            scram_thresholds.append(event_mean)
        
        #calculate mean and 95% CI of scram thresholds
        mean = np.mean(scram_thresholds)
        low_ci,upp_ci = stats.norm.interval(alpha=0.95, loc=mean, \
                                         scale=stats.sem(scram_thresholds))
        
        mean_thresholds.append(mean)
        low_thresholds.append(low_ci)
        upp_thresholds.append(upp_ci)
        
    return mean_thresholds,low_thresholds,upp_thresholds

global years
years = np.array(recon_time)

# Calculate SLP magnitudes
cesm_mean_psl_thresh, cesm_low_psl_thresh, cesm_upp_psl_thresh = \
    calc_thresholds(scram_cesm_recon_psl_nat, start_yrs, stop_yrs)
pace_mean_psl_thresh, pace_low_psl_thresh, pace_upp_psl_thresh = \
    calc_thresholds(scram_pace_recon_psl_nat, start_yrs, stop_yrs)
    
# Calculate U10 magnitudes
cesm_mean_u10_thresh, cesm_low_u10_thresh, cesm_upp_u10_thresh = \
    calc_thresholds(scram_cesm_recon_u10_nat, start_yrs, stop_yrs)
pace_mean_u10_thresh,pace_low_u10_thresh,pace_upp_u10_thresh = \
    calc_thresholds(scram_pace_recon_u10_nat, start_yrs, stop_yrs)

    
#%% 4. Get PI control simulation, split into 86-yr-long members, and put in anomaly space

#concatenates files for LENS PI , which is one long run
pi_ctrl_psl = pi_ctrl.concat_ens_1d('psl',region)
pi_ctrl_u10 = pi_ctrl.concat_ens_1d('u10',region) 
pi_ctrl_time = pi_ctrl.time

# split into 86-year long "members" so all datasets are handled consistently
# also put into anomaly space relative to 41st to 70th values in each "member" (like 1961-1990)
n_pi_ens = int(np.round(len(pi_ctrl_time)/86))
pi_ctrl_ens_psl_anom = np.empty((n_pi_ens,86)) #shape (21,86)
pi_ctrl_ens_psl_anom[:] = np.nan #populate with nans so that uneven num in last ens is nan
pi_ctrl_ens_u10_anom = np.empty((n_pi_ens,86))
pi_ctrl_ens_u10_anom[:] = np.nan

for i in range(n_pi_ens):
    
    #select corresponding data for this ensemble number
    pi_ctrl_psl_ens_i = pi_ctrl_psl[i*86:i*86+86]
    pi_ctrl_u10_ens_i = pi_ctrl_u10[i*86:i*86+86]
    #get anom value to remove in next step
    psl_anom_ref_i = np.mean(pi_ctrl_psl_ens_i[41:71])
    u10_anom_ref_i = np.mean(pi_ctrl_u10_ens_i[41:71])
    
    #populate empty arrays based on size of ensemble
    try:
        pi_ctrl_ens_psl_anom[i,:] = pi_ctrl_psl_ens_i - psl_anom_ref_i
        pi_ctrl_ens_u10_anom[i,:] = pi_ctrl_u10_ens_i - u10_anom_ref_i
    except: #bc last ens doesn't have 86 values (has 81)
        n_last_ens = len(pi_ctrl_psl_ens_i)
        pi_ctrl_ens_psl_anom[i,0:n_last_ens] = pi_ctrl_psl_ens_i - psl_anom_ref_i
        pi_ctrl_ens_u10_anom[i,0:n_last_ens] = pi_ctrl_u10_ens_i - u10_anom_ref_i


#%% 5. Get LENS historical members and put in anom space

hist_mems = lens_hist.members
n_hist_mems = len(hist_mems)

# Load each member, remove mean from 1961-1990 in each member,
# and remove LENS historical EM to get internal component
lens_hist_psl_anom_nat = lens_hist.get_ens_anom_nat('psl', region, anom_ref)
lens_hist_u10_anom_nat = lens_hist.get_ens_anom_nat('u10', region, anom_ref)



#%% 5. Make Figure 5: Plot all datasets

# Plot SLP on left, U on right
# Order data to plot in subplot order (top left, top right, next row...)
times = [recon_time,recon_time,\
         recon_time, recon_time,\
         recon_time, recon_time, \
         np.linspace(0,85,86), np.linspace(0,85,86), \
         lens_hist.time, lens_hist.time]
datasets = [cesm_recon_psl_ens, cesm_recon_u10_ens,\
            scram_cesm_recon_psl_nat, scram_cesm_recon_u10_nat,\
            scram_pace_recon_psl_nat, scram_pace_recon_u10_nat,\
            pi_ctrl_ens_psl_anom, pi_ctrl_ens_u10_anom,\
            lens_hist_psl_anom_nat, lens_hist_u10_anom_nat]
hi_psl_col = '#ab3f50'
pi_psl_col = '#b38184'
hi_u_col = '#674064'
pi_u_col = '#9a7fa9'

    
#set up metadata to iterate through
labels = ['a) CESM LM recon unscrambled ensemble','b) CESM LM unscrambled ensemble',\
          'c) CESM LM recon scrambled ensemble (internal)', 'd) CESM LM recon ensemble (internal)',\
          'e) PACE recon scrambled ensemble (internal)', 'f) PACE recon scrambled ensemble (internal)',\
          'g) Preindustrial control ensemble','h) Preindustrial control ensemble',\
          'i) LENS Historical ensemble (internal)','j) LENS Historical ensemble (internal)']
cesm_recon_psl_colors = plt.cm.Blues(np.linspace(0.3,1,n_draws))
pace_recon_psl_colors = plt.cm.Blues(np.linspace(0,0.8,n_draws))
cesm_recon_u_colors = plt.cm.YlOrBr(np.linspace(0.3,1,n_draws))
pace_recon_u_colors = plt.cm.Oranges(np.linspace(0,0.8,n_draws))
colors = [cesm_recon_psl_colors[::3], cesm_recon_u_colors[::3],\
          cesm_recon_psl_colors, cesm_recon_u_colors,\
          pace_recon_psl_colors, pace_recon_u_colors,\
          pi_psl_col,pi_u_col, \
          hi_psl_col,hi_u_col]

fs = 7
lw = 0.4
fig = plt.figure()
fig.set_size_inches(6.5,6.5)
axes = []
for i in range(10):
    
    
    ax = fig.add_subplot(6,2,i+1)
    ax.tick_params(direction="in")
    dataset = datasets[i]
    if i == 0 or i ==1:
        dataset = np.transpose(np.array(dataset))
    time = times[i]
    lab = labels[i]
    color = colors[i]
    for ens in range(len(dataset)):
        
        # Plot model data and scrambled recons
        if type(color) == str:
            col = color
        else:
            col = color[ens]
        try:
            ax.plot(time,dataset[ens],linewidth=lw,color=col)
        # Plot unscrambled recons
        except:
            ax.plot(time,dataset[:,ens],linewidth=lw,color=col)
    ax.annotate(lab,(0.02,.86),xycoords = 'axes fraction',\
                fontsize=fs,weight='bold')    
    
    #for unscrambled ens, annotate variance of one member
    if i ==0 or i ==1:
        data_var = np.var(dataset[ens,:])
        ax.annotate('Variance of each member = {:.2f}'.format(data_var),\
              (.02,.08), xycoords='axes fraction',fontsize=fs) 
    #for all other data, plot variance of all members (flattens members)
    else:
        data_var = np.nanvar(dataset)
        ax.annotate('Variance = {:.2f}'.format(data_var),\
              (.02,.08), xycoords='axes fraction',fontsize=fs) 
    axes.append(ax)
    
axes[0].set_title('SLP anomaly (hPa)',fontsize = fs+2)
axes[1].set_title(r'U$_S$ anomaly (m/s)',fontsize=fs+2)

# Set SLP lims
for ax in axes[::2]:
    ax.set_ylim([-11,14])
    ax.set_yticks([-8,0,8])
# Set u limes
for ax in axes[1::2]:
    ax.set_ylim([-3.5,4])
    ax.set_yticks([-2,0,2])
    

#plot LENS EM SLP
ax11 = fig.add_subplot(6,2,11)
ax11.plot(lens_hist.time,lens_hist_em_psl_anom,color='#777777')
ax11.annotate('k) LENS historical ensemble mean',(0.02,.86),xycoords = 'axes fraction',\
            fontsize=fs,weight='bold') 
ax11.set_ylim([-2.6,2.6])
ax11.set_yticks([-2,0,2])
ax11.set_xlabel('Year')

#plot LENS EM U
ax12 = fig.add_subplot(6,2,12)
ax12.plot(lens_hist.time,lens_hist_em_u10_anom,color='#555555')
ax12.annotate('l) LENS historical ensemble mean',(0.02,.86),xycoords = 'axes fraction',\
            fontsize=fs,weight='bold')   
ax12.set_ylim([-.7,.8])
ax12.set_yticks([-0.5,0,0.5])
ax12.yaxis.set_tick_params(pad=-.25)
ax12.set_xlabel('Year')
    
plt.subplots_adjust(left=0.04,right=0.99,top=0.96,bottom=0.06,wspace=0.1)
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
    
if save_fig:
    plt.savefig('Plots/Figure_5_timeseries_for_search.png',dpi=600)
        

#%% 6. Search model for occurrences exceeding recon magnitudes

#Using an m-year sliding window so you are looking at n_model_years - m possible m-yr events
#(I.e. if you have 1000 years, you search 999 2-year events)

#We want to minimize overlap to avoid autocorrelated samples. 
#Slide each window over by m//2 years, rather than 1 year.

def search_model(model_data,events,thresholds):
    """
    For a given set of n thresholds corresponding to n m-yr long events, 
    search a model time series for m-year-long events meeting threshold.
    Using a sliding window with 50% overlap.

    Parameters
    ----------
    model_data : 1d numpy array
        Model timeseries data
    events : 1d numpy array 
        how many years are in each event
    thresholds : 1d numpy array
        thresholds for each event, same order as events. 
        i.e. mean thresholds or lower CI thresholds.

    Returns
    -------
    all_event_counters : list
        number of events found in model for each event type
        follows same order as events
    n_samples : list of ints of len n_events
        numbers of windows checked

    """
    
    #will make list of len n_events
    all_event_counters = []
    n_samples = []
    event_mags = [] #will have 10 lists, each list has len corresponding to event sample size with value = event magnitudes
    
    #search for each type of event
    for i in range(len(events)):
        
        m_yrs = int(events[i])
        #print('Searching ',m_yrs,'-yr events...')
        sample_size = 0
        
        event_counter = 0
        event_thresh = thresholds[i]
        event_i_mags = []
        
        #search in each sliding window possible with step size m//2
        j = 0
        while j <= len(model_data) - m_yrs:
            
            sample_size += 1
            
            event_j = model_data[j:j+m_yrs]
            #print last checks
            #if j > 1789:
                #print('Checking event from: ',lens_pi_time[j:j+m_yrs],'\n')
            #make sure event is long enough
            
            event_j_mean = np.mean(event_j)
            #add magnitude of event if not nan
            if not np.isnan(event_j_mean):
                event_i_mags.append(event_j_mean)
            if event_j_mean >= event_thresh:
                event_counter += 1
            
            #increase indx by 50% of window, rounding up for odd event nums
            j += int(np.ceil(m_yrs/2)) 
        
        # print(event_counter,' events found!')
        all_event_counters.append(event_counter)
        n_samples.append(sample_size)
        event_mags.append(event_i_mags)
    
    return all_event_counters, n_samples, event_mags



# iterate through all thresholds in a specified model using specified recon  for a specific variable
def run_search_and_calc_event_mags(model_data,model_name,recon_thresholds):
    """
    This should be run for all combos of model types, variables, and reconstructions.
    It handles the 10 event types and 3 levels of reconstruction magnitudes (low, mean, upper CI).
    
    For a specified model dataset (i.e. LENS PI SLP), iterature through
    all 3 threshold levels for a given reconstruction (i.e. CESM LM SLP thresholds)
    to keep track of the events found in the model (calls function search_model).
    
    Returns number of occurrences found in the model, for each event type (1-10 years)
    for each of 3 threshold levels. Also return that as a return period per 10ka
    (the same as the event count, but scaled up based on sample size found in model)
    
    Also return the magnitudes of all event types in the model (for calculating sigma)

    Parameters
    ----------
    model_data : 1d numpy array
        contains SLP or U data in a specified model, used for conducting search.
        example: lens_pi_ens_psl_anom
    model_name : str
        describes model for special handling of the LENS PI model, which contains nans
        due to uneven split into 86-year long ensembles
        accepted: 'PI_ctrl' or other
    recon_thresholds : list of 3 items, each containing a list of 10 floats
        Contains the 1940 event magnitudes (the lower, mean, and upper values)
        found in the scrambled reconstruction.
        Corresponds to output from calc_thresholds function above.

    Returns
    -------
    event_count_all_thresholds : list of 3 arrays each containing 10 floats
        3 arrays correspond to low, mean, and upper thresholds
        each array contains number of events found in model that exceed the corresponding threshold
        10 floats correspond to events lasting 1-10 years
    event_count_10ka_all_thresholds : list of 3 lists, each containing10 floats
        same as event_count_all_thresholds, but scaled up to reflect events per 10ka
        scaling is based on the sample size in the model (varies based on event length)
    model_event_mags_dict : dictionary with 10 keys corresponding to 10 events
        contains the magnitudes of each type of event found in the model
        independent of the recon magnitudes

    """
    
    n_ens = len(model_data)
    event_count_all_thresholds = [] #will be a list of 3 lists each containing 10 event counts
    event_count_10ka_all_thresholds = [] #will also be a list of 3 lists
    
    #create dict of all model magnitudes for each event (only requires loping through ensembles)
    model_event_mags_dict = {} #independent of thresholds, so only update in first iteration of thresholds
    
    #for each threshold (low,mean,high)
    for thresh in recon_thresholds:
        
        #start counter for this threshold
        event_count_thresh = np.zeros(len(thresh),) 
    
        #iterate through each model ensemble member
        for ens in range(n_ens):
        
            #find events with that magnitude in that model
            n_events, n_samples, event_mags = search_model(model_data[ens],\
                                                    events, thresh)
            #update event counter for this threshold
            event_count_thresh = np.add(event_count_thresh,n_events)
            
            #Update dictionary with all event magnitudes found in model (if first iteration, since only need to do once)
            if thresh == recon_thresholds[0]:
                
                for event in events:
                    
                    #if entries already exist, append new magnitudes to the entry
                    try:
                        event_mags_old = model_event_mags_dict[event]
                        event_mags_new = event_mags_old + event_mags[event - 1] #use index version of event int
                        model_event_mags_dict[event] = event_mags_new
                        
                    #create new entry if empty
                    except:
                        model_event_mags_dict[event] = event_mags[event-1] #Fixed error here!!! was adding last one (was "event-2")
                        
        event_count_all_thresholds.append(event_count_thresh)
    
    
        #calculate return period for x events in 10ka
        #number of samples for 1-yr long events in each ens is 86. decreases quickly with 50% overlap windows.
        n_samples_total = np.multiply(n_samples,n_ens)
        if 'PI_ctrl' in model_name:
            n_samples_total[0] = 1801 #since last ens has only 81 values, multiplying produces too many samples for the 1-yr events
        event_count_10ka_thresh = []
        
        #go through each of 10 events using counts for this threshold level
        for n_hits,n_samp in zip(event_count_thresh,n_samples_total):
            scalar = 10000/n_samp
            event_count_10ka_event = n_hits * scalar
            event_count_10ka_thresh.append(int(event_count_10ka_event))
            
        event_count_10ka_all_thresholds.append(event_count_10ka_thresh)
        
    return event_count_all_thresholds, event_count_10ka_all_thresholds, model_event_mags_dict


#Perform search for each model for each recon for each var 

cesm_thresholds_psl = [cesm_low_psl_thresh, cesm_mean_psl_thresh, cesm_upp_psl_thresh]
pace_thresholds_psl = [pace_low_psl_thresh, pace_mean_psl_thresh, pace_upp_psl_thresh]
cesm_thresholds_u10 = [cesm_low_u10_thresh, cesm_mean_u10_thresh, cesm_upp_u10_thresh]
pace_thresholds_u10 = [pace_low_u10_thresh, pace_mean_u10_thresh, pace_upp_u10_thresh]

#Output is in order of low thresh, mean, high thresh

#for LENS PI simulation:
pi_cesm_psl_event_cts, pi_cesm_psl_event_cts_10ka, pi_psl_event_mags = \
    run_search_and_calc_event_mags(pi_ctrl_ens_psl_anom, 'PI_ctrl', cesm_thresholds_psl)
    
pi_pace_psl_event_cts, pi_pace_psl_event_cts_10ka, pi_psl_event_mags = \
    run_search_and_calc_event_mags(pi_ctrl_ens_psl_anom, 'PI_ctrl', pace_thresholds_psl)
    
pi_cesm_u10_event_cts, pi_cesm_u10_event_cts_10ka, pi_u10_event_mags = \
    run_search_and_calc_event_mags(pi_ctrl_ens_u10_anom, 'PI_ctrl', cesm_thresholds_u10)
    
pi_pace_u10_event_cts, pi_pace_u10_event_cts_10ka, pi_u10_event_mags = \
    run_search_and_calc_event_mags(pi_ctrl_ens_u10_anom, 'PI_ctrl', pace_thresholds_u10)
    
#for LENS HI simulation:
hi_cesm_psl_event_cts, hi_cesm_psl_event_cts_10ka, hi_psl_event_mags = \
    run_search_and_calc_event_mags(lens_hist_psl_anom_nat, 'LENS Historical', cesm_thresholds_psl)
    
hi_pace_psl_event_cts, hi_pace_psl_event_cts_10ka, hi_psl_event_mags = \
    run_search_and_calc_event_mags(lens_hist_psl_anom_nat, 'LENS Historical', pace_thresholds_psl)
    
hi_cesm_u10_event_cts, hi_cesm_u10_event_cts_10ka, hi_u10_event_mags = \
    run_search_and_calc_event_mags(lens_hist_u10_anom_nat, 'LENS Historical', cesm_thresholds_u10)
    
hi_pace_u10_event_cts, hi_pace_u10_event_cts_10ka, hi_u_event_mags = \
    run_search_and_calc_event_mags(lens_hist_u10_anom_nat, 'LENS Historical', pace_thresholds_u10)
    


#%% 7. Make table with event occurences


def make_event_table(vname,pi_cesm_data,hi_cesm_data,pi_pace_data,hi_pace_data):
    
    fig, ax = plt.subplots()
    height = 5
    fig.set_size_inches(7,height)
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    
    #select mean values of data with lower bound in parentheses
    text = []
    for i in range(n_events):
        
        #1st index is the mean, 1nd is the count based on the upper mag (lower bound of count)
        pi_cesm_text = str(pi_cesm_data[1][i]) + ' ('+str(pi_cesm_data[2][i])+')'
        hi_cesm_text = str(hi_cesm_data[1][i]) + ' ('+str(hi_cesm_data[2][i])+')'
        
        pi_pace_text = str(pi_pace_data[1][i]) + ' ('+str(pi_pace_data[2][i])+')'
        hi_pace_text = str(hi_pace_data[1][i]) + ' ('+str(hi_pace_data[2][i])+')'
        
        text_event_i = [pi_cesm_text,hi_cesm_text,pi_pace_text,hi_pace_text]
        text.append(text_event_i)
        
    columns = ['PI Ctrl\nCESM recon','LENS Hist Int\nCESM recon',\
                'PI Ctrl\nPACE recon','LENS Hist Int\nPACE recon'] 
    tb = ax.table(cellText=text,
                          rowLabels=events,
                          #rowColours=colors,
                          loc='center',
                          colLabels=columns)
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    table_props = tb.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells: 
        cell.set_height((height-4)/n_events)
    plt.suptitle(vname)
        
    return

make_event_table('SLP', pi_cesm_psl_event_cts_10ka, hi_cesm_psl_event_cts_10ka,\
                 pi_pace_psl_event_cts_10ka, hi_pace_psl_event_cts_10ka)
    
make_event_table('Us', pi_cesm_u10_event_cts_10ka, hi_cesm_u10_event_cts_10ka,\
                 pi_pace_u10_event_cts_10ka, hi_pace_u10_event_cts_10ka)

#%% Calculate z scores to evaluate significance of events

# First check that all data are ~ normally distributed
#check that data are normally distributed so you can do z test to determine significance of 1940 event

# def normal_test(data,alpha,data_lab):
    
    
#     fig = plt.figure()
#     plt.hist(data)
#     plt.title(data_lab)

#     # stat, p = stats.normaltest(data)
#     # stat, p = stats.shapiro(data)
#     k2, p = stats.kstest(data,'norm')
#     # null hypothesis: x comes from a normal distribution
#     if p < alpha:  #null hypothesis can be rejected
#         print(data_lab+" not normal. p =",p)
#     else: #null hypothesis cannot be rejected
#         print(data_lab+" is probably normal , p =",p)
        

# alpha = 0.05
# for event in pi_psl_event_mags_dict.keys():
#     normal_test(event,alpha,event)
    
# normal_test(lens_pi_ens_psl_anom.flatten()[0:1801], alpha, 'LENS PI SLP') #get rid of nans at end
# normal_test(lens_hist_psl_anom_nat.flatten(), alpha, 'LENS HI SLP dt')
# normal_test(lens_pi_ens_u_anom.flatten()[0:1801], alpha, 'LENS PI U')
# normal_test(lens_hist_u_anom_nat.flatten(), alpha, 'LENS HI U')

#They are normal by plotting histograms of the data. which all show a bell curve 

#%% 8. Calculate z scores

def calc_z_score(x,comp_data):
    
    z_score = (x - np.mean(comp_data)) / np.std(comp_data)
    
    return z_score
    

# Get sigma levels for all event types
pi_cesm_psl_z_all_events = [] #list of 10 items, each item contains 3 elements ordered PI, HI
pi_pace_psl_z_all_events = []
hi_cesm_psl_z_all_events = [] 
hi_pace_psl_z_all_events = []
pi_cesm_u10_z_all_events = []
pi_pace_u10_z_all_events = []
hi_cesm_u10_z_all_events = []
hi_pace_u10_z_all_events = []

for i in range(n_events):
    #for each event use i for index, use i + 1 for dictionary 
    
    #SLP with mean thresholds
    pi_cesm_psl_z = calc_z_score(cesm_mean_psl_thresh[i],pi_psl_event_mags[i+1])
    pi_pace_psl_z = calc_z_score(pace_mean_psl_thresh[i],pi_psl_event_mags[i+1])
    hi_cesm_psl_z = calc_z_score(cesm_mean_psl_thresh[i],hi_psl_event_mags[i+1])
    hi_pace_psl_z = calc_z_score(pace_mean_psl_thresh[i],hi_psl_event_mags[i+1])
    pi_cesm_psl_z_all_events.append(pi_cesm_psl_z)
    pi_pace_psl_z_all_events.append(pi_pace_psl_z)
    hi_cesm_psl_z_all_events.append(hi_cesm_psl_z)
    hi_pace_psl_z_all_events.append(hi_pace_psl_z)

        
    #U with mean thresholds
    pi_cesm_u10_z = calc_z_score(cesm_mean_u10_thresh[i],pi_u10_event_mags[i+1])
    pi_pace_u10_z = calc_z_score(pace_mean_u10_thresh[i],pi_u10_event_mags[i+1])
    hi_cesm_u10_z = calc_z_score(cesm_mean_u10_thresh[i],hi_u10_event_mags[i+1])
    hi_pace_u10_z = calc_z_score(pace_mean_u10_thresh[i],hi_u10_event_mags[i+1])
    pi_cesm_u10_z_all_events.append(pi_cesm_u10_z)
    pi_pace_u10_z_all_events.append(pi_pace_u10_z)
    hi_cesm_u10_z_all_events.append(hi_cesm_u10_z)
    hi_pace_u10_z_all_events.append(hi_pace_u10_z)

        
    


#%% 9. Make Figures 6 and 7: plot recon magnitudes, sigmas, and num of occurrences in models

cesm_run_psl_col = '#1f3a6f'
cesm_run_u_col = 'chocolate'
pace_run_psl_col = '#476ebb'
pace_run_u_col = '#ffb752'

def make_probability_fig(vname,cesm_thresh,pace_thresh,\
                         cesm_z_scores,pace_z_scores,\
                         pi_cesm_event_cts_10ka, hi_cesm_event_cts_10ka,\
                         pi_pace_event_cts_10ka, hi_pace_event_cts_10ka):
    
    low_cesm_thresh,mean_cesm_thresh,upp_cesm_thresh = cesm_thresh
    low_pace_thresh,mean_pace_thresh,upp_pace_thresh = pace_thresh
    
    
    if vname == 'psl':
        cesm_run_col = cesm_run_psl_col
        pace_run_col = pace_run_psl_col
        hi_col = hi_psl_col
        pi_col = pi_psl_col
        units = '(hPa)'
    else:
        cesm_run_col = cesm_run_u_col
        pace_run_col = pace_run_u_col
        hi_col = hi_u_col
        pi_col = pi_u_col
        units = '(m/s)'
    
    fig = plt.figure()
    fig.set_size_inches(5.75,4.45)
    spec = gridspec.GridSpec(ncols=2, nrows=3,wspace=0.23,
                             hspace=0.2, height_ratios=[1, 1.2, 1.2])
    #adjust figure properties
    f1,f2 = 6,8
    ms = 3 #marker size
    cs = 4 #capsize #was 5 for 10 events
    space = 0.18 #space between different model points
    plt.rc('font', size=f1)          # controls default text sizes
    plt.rc('axes', titlesize=f2)     # fontsize of the axes title
    plt.rc('axes', labelsize=f2)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=f1)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=f1)    # fontsize of the tick labels
    plt.rc('legend', fontsize=f2)    # legend fontsize
    plt.rc('figure', titlesize=f2)  # fontsize of the figure title
    
    #Top row: plot thresholds---------------------------------------------------
    
    ax0 = fig.add_subplot(spec[0])
    ax0.grid(alpha=.5,axis='y')
    [plt.axvline(x,linewidth=0.5,alpha=0.5,color='gray') for x in events-.5]
    lower_err = mean_cesm_thresh-low_cesm_thresh
    upper_err = upp_cesm_thresh-mean_cesm_thresh
    error = np.array(list(zip(lower_err,upper_err))).T
    ax0.errorbar(events,mean_cesm_thresh,fmt='o',yerr=error,capsize=cs,\
                 color=cesm_run_col,ms=ms)
    ax0.set_title('CESM LM reconstruction')
    
    if vname == 'psl':
        plt.ylim([1.2,5.1])
        plt.ylabel('Magnitude '+units,labelpad=8.5)
    else:
        plt.ylim([0.25,1.6])
        plt.ylabel('Magnitude '+units,labelpad=3)
    plt.annotate('a)',(0.01,.86),xycoords = 'axes fraction',fontsize=fs+2,weight='bold')
    
    
    fig.add_subplot(spec[1])
    plt.grid(alpha=.5,axis='y')
    [plt.axvline(x,linewidth=0.5,alpha=0.5,color='gray') for x in events-.5]
    lower_err = mean_pace_thresh-low_pace_thresh
    upper_err = upp_pace_thresh-mean_pace_thresh
    error = np.array(list(zip(lower_err,upper_err))).T
    plt.errorbar(events,mean_pace_thresh,fmt='o',yerr=error,capsize=cs,\
                 color=pace_run_col,ms=ms)
    plt.xticks(events)
    plt.xlim([-0.5,n_events+.5])
    plt.title('PACE reconstruction')
    if vname == 'psl':
        plt.ylim([1.2,5.1])
        plt.ylabel('Magnitude '+units,labelpad=10.5)
    else:
        plt.ylim([0.25,1.6])
        plt.ylabel('Magnitude '+units,labelpad=4)
    plt.annotate('b)',(0.01,.86),xycoords = 'axes fraction',fontsize=fs+2,weight='bold')
    
    #Middle row: plot event sigma levels----------------------------------------
    
    ax1 = fig.add_subplot(spec[2])
    #yerr = 0.1 for all events, based on z score analysis above
    ax1.errorbar(events-space,cesm_z_scores[0],yerr=0.1,label='PI Ctrl',color=pi_col,fmt='o',capsize=cs-2.5,ms=ms)
    ax1.errorbar(events+space,cesm_z_scores[1],yerr=0.1,label='LENS Hist Int',color=hi_col,fmt='o',capsize=cs-2.5,ms=ms)
    plt.grid(alpha=.5,axis='y')
    [plt.axvline(x,linewidth=0.5,alpha=0.5,color='gray') for x in events-.5]
    ax1.set_xlim([-0.5,n_events+.5])
    plt.xticks(events)
    ax1.hlines(2,0.25,n_events+0.25,color='gray',linestyle='--')
    if vname == 'psl':
        ax1.set_ylim([1,3])
        ax1.set_ylabel('Sigma',labelpad=4)
    else:
        ax1.set_ylim([1,3.4])
        ax1.set_ylabel('Sigma',labelpad=6)
    ax1.legend(loc='lower right',ncol=2,fontsize=8,handletextpad=0.1)
    ax1.annotate('c)',(0.01,.88),xycoords = 'axes fraction',fontsize=fs+2,weight='bold')
    
    ax2 = fig.add_subplot(spec[3])
    ax2.errorbar(events-space,pace_z_scores[0],yerr=0.1,label='PI Ctrl',color=pi_col,fmt='o',capsize=cs-2.5,ms=ms)
    ax2.errorbar(events+space,pace_z_scores[1],yerr=0.1,label='LENS Hist Int',color=hi_col,fmt='o',capsize=cs-2.5,ms=ms)
    plt.grid(alpha=.5,axis='y')
    [plt.axvline(x,linewidth=0.5,alpha=0.5,color='gray') for x in events-.5]
    ax2.set_xlim([-0.5,n_events+.5])
    plt.xticks(events)
    ax2.hlines(2,0.25,n_events+0.25,color='gray',linestyle='--')
    if vname == 'psl':
        ax2.set_ylim([1,3])
        ax2.set_ylabel('Sigma',labelpad=7)
    else:
        ax2.set_ylim([1,3.4])
        ax2.set_ylabel('Sigma',labelpad=8)
    ax2.legend(loc='upper right',ncol=2,fontsize=8,handletextpad=0.1)
    ax2.annotate('d)',(0.01,.88),xycoords = 'axes fraction',fontsize=fs+2,weight='bold')
    
    
    #bottom row: plot event occurences --------------------------------------------------
    
    width = 0.25 #width of histograms
    ms = .5 #marker size of mean value
    cap = 3 #length of error caps
    ewidth = 1 #width of vertical line in error bars
    
    #Plot CESM occurrences
    ax3 = fig.add_subplot(spec[4])
    #LENS PI
    ax3.bar(events-space,pi_cesm_event_cts_10ka[1],label='PI Ctrl',width=width,color=pi_col)
    lower_pi_err = np.array(pi_cesm_event_cts_10ka[1]) - np.array(pi_cesm_event_cts_10ka[0])
    upper_pi_err = np.array(pi_cesm_event_cts_10ka[2]) - np.array(pi_cesm_event_cts_10ka[1])
    pi_cesm_err = np.array(list(zip(lower_pi_err,upper_pi_err))).T
    ax3.errorbar(events-space,pi_cesm_event_cts_10ka[1],fmt='o',yerr=pi_cesm_err,capsize=cap,\
                 color='k',elinewidth=ewidth,ms=ms)
    #LENS hist
    ax3.bar(events+space,hi_cesm_event_cts_10ka[1],label='LENS Hist Int',\
            width=width,color=hi_col)
    lower_hist_err = np.array(hi_cesm_event_cts_10ka[1]) - np.array(hi_cesm_event_cts_10ka[0])
    upper_hist_err = np.array(hi_cesm_event_cts_10ka[2]) - np.array(hi_cesm_event_cts_10ka[1])
    hist_cesm_err = np.array(list(zip(lower_hist_err,upper_hist_err))).T
    ax3.errorbar(events+space,hi_cesm_event_cts_10ka[1],fmt='o',yerr=hist_cesm_err,\
                 capsize=cap, color='k',ms=ms,elinewidth=ewidth)
    
    ax3.set_ylim([0,225])
    ax3.legend(fontsize=8,loc='upper right',ncol=2)
    ax3.grid(linewidth = 0.5,color='gray',axis='y',alpha=0.5)
    if vname == 'psl':
        ax3.set_ylabel('Occurrences per 10ka',labelpad=1.25)
    else:
        ax3.set_ylabel('Occurrences per 10ka',labelpad=2)
    ax3.set_xlabel('Number of years in event',labelpad=1)
    ax3.set_xticks(events)
    ax3.set_xlim([-0.5,n_events+.5])
    ax3.annotate('e)',(0.01,.86888),xycoords = 'axes fraction',fontsize=fs+2,weight='bold')
    
    
    #plot PACE occurrences----------------------------------------
    
    ax5 = fig.add_subplot(spec[5])
    #LENS PI
    ax5.bar(events-space,pi_pace_event_cts_10ka[1],label='PI Ctrl',width=width,color=pi_col)
    lower_pi_err = np.array(pi_pace_event_cts_10ka[1]) - np.array(pi_pace_event_cts_10ka[0]) 
    upper_pi_err = np.array(pi_pace_event_cts_10ka[2]) - np.array(pi_pace_event_cts_10ka[1])
    pi_pace_err = np.array(list(zip(lower_pi_err,upper_pi_err))).T
    ax5.errorbar(events-space,pi_pace_event_cts_10ka[1],fmt='o',yerr=pi_pace_err,capsize=cap,\
                 color='k',elinewidth=ewidth,ms=ms)
    #LENS hist
    ax5.bar(events+space,hi_pace_event_cts_10ka[1],label='LENS Hist Int',\
            width=width,color=hi_col)
    lower_hist_err = np.array(hi_pace_event_cts_10ka[1]) - np.array(hi_pace_event_cts_10ka[0])
    upper_hist_err = np.array(hi_pace_event_cts_10ka[2]) - np.array(hi_pace_event_cts_10ka[1])
    hist_pace_err = np.array(list(zip(lower_hist_err,upper_hist_err))).T
    ax5.errorbar(events+space,hi_pace_event_cts_10ka[1],fmt='o',yerr=hist_pace_err,\
                 capsize=cap, color='k',ms=ms,elinewidth=ewidth)
    
    ax5.set_ylim([0,1200])
    ax5.grid(linewidth = 0.5,color='gray',axis='y',alpha=0.5)
    if vname == 'psl':
        ax5.set_ylabel('Occurrences per 10ka',labelpad=-.5)
    else:
        ax5.set_ylabel('Occurrences per 10ka',labelpad=1)
    ax5.set_xlabel('Number of years in event',labelpad=1)
    ax5.set_xticks(events)
    ax5.set_xlim([-0.5,n_events+.5])
    ax5.legend(fontsize=8,loc='upper right',ncol=2)
    ax5.annotate('f)',(0.02,.88),xycoords = 'axes fraction',fontsize=fs+2,weight='bold')
    
    axes = [ax0,ax1,ax2,ax3,ax4,ax5,ax6]
    for ax in axes:
        ax.set_xticks(events)
        ax.set_xlim([-0.5,n_events+.5])
    #-----------
    
    
    plt.subplots_adjust(top=.95,left=.08,right=.995,bottom = 0.08,) #can't adjust wspace, hspace using gridspec
    
    return


cesm_thresholds_psl = [np.array(x) for x in cesm_thresholds_psl]
pace_thresholds_psl = [np.array(x) for x in pace_thresholds_psl]
cesm_psl_z_scores = [pi_cesm_psl_z_all_events, hi_cesm_psl_z_all_events]
pace_psl_z_scores = [pi_pace_psl_z_all_events, hi_pace_psl_z_all_events]


make_probability_fig('psl', cesm_thresholds_psl, pace_thresholds_psl,\
                     cesm_psl_z_scores, pace_psl_z_scores,\
                     pi_cesm_psl_event_cts_10ka, hi_cesm_psl_event_cts_10ka,\
                     pi_pace_psl_event_cts_10ka, hi_pace_psl_event_cts_10ka)
if save_fig:
    plt.savefig('Plots/Figure_6_SLP_thresholds_event_occurrences.png',dpi=600)



cesm_thresholds_u10 = [np.array(x) for x in cesm_thresholds_u10]
pace_thresholds_u10 = [np.array(x) for x in pace_thresholds_u10]
cesm_u10_z_scores = [pi_cesm_u10_z_all_events, hi_cesm_u10_z_all_events]
pace_u10_z_scores = [pi_pace_u10_z_all_events, hi_pace_u10_z_all_events]


make_probability_fig('u10', cesm_thresholds_u10, pace_thresholds_u10,\
                     cesm_u10_z_scores, pace_u10_z_scores,\
                     pi_cesm_u10_event_cts_10ka, hi_cesm_u10_event_cts_10ka,\
                     pi_pace_u10_event_cts_10ka, hi_pace_u10_event_cts_10ka)

if save_fig:
    plt.savefig('Plots/Figure_7_U_thresholds_event_occurrences.png',dpi=600)
    
    
