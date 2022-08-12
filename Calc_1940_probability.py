#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:39:39 2022

Calculating 1940 Probabilty using methods discussed in Japan with Paul and Greg. 
Removing EM from LENS model and recon, and calculating probabilities
for 1-10 year long events. 

Change since v4: moving the LENS EM so that it matches ERA5. The mean for SLP
is quite off, which made a mismatch between LENS PI and the internal component
of LENS historical. Shifting it should improve the accuracy of isolating
the internal component. 

Copied Calc_1940_prob.py from LMR_analysis directory and starting making
changes using git starting 8/12/22.

Run it chunk by chunk to check for parameters in each section.

@author: gemma
"""

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import scipy.stats as stats
from Functions_1940_analysis import load_1d_recon_full_ens, load_1d_recon,\
    load_1d_clim_model_concat_ens,load_1d_clim_model_em,calc_conf_int_ind_data,\
    load_1d_era5
    
#%%
#GET RAW MODEL DATA

lens_pi_time,lens_pi_u = load_1d_clim_model_concat_ens('LENS_pi','u10','ASE') 
lens_pi_time,lens_pi_psl = load_1d_clim_model_concat_ens('LENS_pi','psl','ASE') 
lens_hist_time,lens_hist_u = load_1d_clim_model_concat_ens('LENS_hist','u10','ASE') 
lens_hist_time,lens_hist_psl = load_1d_clim_model_concat_ens('LENS_hist','psl','ASE') 
lens_em_time,lens_em_u = load_1d_clim_model_em('LENS_hist','u10','ASE')
lens_em_time,lens_em_psl = load_1d_clim_model_em('LENS_hist','psl','ASE')


plot_raw_model_data = False

if plot_raw_model_data:
    fig=plt.figure()
    
    fig.add_subplot(321)
    plt.plot(lens_pi_time,lens_pi_u,label='LENS PI',color='gray',linewidth=.7)
    plt.ylabel('Us (m/s)')
    plt.legend(ncol=2)
    
    fig.add_subplot(323)
    plt.plot(lens_hist_time,lens_hist_u,label='LENS Historical concatenated members',color='lightsteelblue',linewidth=.7)
    plt.plot(lens_em_time,lens_em_u,label='LENS Historical ensemble mean',color='tab:blue',linewidth=.7)
    plt.ylabel('Us (m/s)')
    plt.legend(ncol=1)
    
    fig.add_subplot(325)
    plt.plot(lens_em_time,lens_em_u,label='LENS Historical ensemble mean',color='tab:blue',linewidth=.7)
    plt.ylabel('Us (m/s)')
    plt.legend(ncol=2)
    
    fig.add_subplot(322)
    plt.plot(lens_pi_time,lens_pi_psl,label='LENS PI',color='gray',linewidth=.7)
    plt.ylabel('SLP (hPa)')
    plt.legend(ncol=2)
    
    fig.add_subplot(324)
    plt.plot(lens_hist_time,lens_hist_psl,label='LENS Historical concatenated members',color='lightsteelblue',linewidth=.7)
    plt.plot(lens_em_time,lens_em_psl,label='LENS Historical ensemble mean',color='tab:blue',linewidth=.7)
    plt.ylabel('SLP (hPa)')
    plt.legend(ncol=1)
    
    fig.add_subplot(326)
    plt.plot(lens_em_time,lens_em_psl,label='LENS Historical ensemble mean',color='tab:blue',linewidth=.7)
    plt.ylabel('SLP (hPa)')
    plt.legend(ncol=2)
    
    plt.suptitle('Raw model data in ASE')
    
    plt.suptitle('Raw model ASE shelf break winds')

#%% GET ERA5 data and shift LENS EM so that it matches

era_time,era_psl = load_1d_era5('psl',[1979,2005],'ASE')
era_time,era_u = load_1d_era5('u1000',[1979,2005],'ASE')

#calculate offset in mean between ERA and LENS EM from 1979-2005
psl_os = np.mean(era_psl) - np.mean(lens_em_psl.sel(time=slice(1979,2005))) #2.25
u_os = np.mean(era_u) - np.mean(lens_em_u.sel(time=slice(1979,2005))) #0.05

#shift LENS EM by those offsets
lens_em_psl_adj = lens_em_psl + psl_os
lens_em_u_adj = lens_em_u + u_os

#%% Remove LENS EM adj from LENS historical and put LENS PI in anomaly space

#remove LENS EM from LENS historical to isolate internal component
lens_hist_time,lens_hist_u_nat = load_1d_clim_model_concat_ens('LENS_hist',\
                                'u10','ASE',isolate_nat=True,adj_lens_em = True) 
lens_hist_time,lens_hist_psl_nat = load_1d_clim_model_concat_ens('LENS_hist',\
                                'psl','ASE',isolate_nat=True,adj_lens_em=True) 

    
#remove LENS EM from 1961-1990 in LENS PI so it is in the same anomaly space
lens_em_u_adj_ref = np.mean(lens_em_u_adj.sel(time = slice(1961,1990)))
lens_pi_u_anom = lens_pi_u - np.tile(lens_em_u_adj_ref,len(lens_pi_u))
lens_em_psl_adj_ref = np.mean(lens_em_psl_adj.sel(time = slice(1961,1990)))
lens_pi_psl_anom = lens_pi_psl - np.tile(lens_em_psl_adj_ref,len(lens_pi_psl))


#also remove LENS EM ref value from LENS hist for the calculation with LENS hist including anthro
lens_em_u_adj_ref = np.mean(lens_em_u_adj.sel(time = slice(1961,1990)))
lens_hist_u_adj_anom = lens_hist_u - np.tile(lens_em_u_adj_ref,len(lens_hist_u))
lens_em_psl_adj_ref = np.mean(lens_em_psl_adj.sel(time = slice(1961,1990)))
lens_hist_psl_adj_anom = lens_hist_psl - np.tile(lens_em_psl_adj_ref,len(lens_hist_psl))


#put LENS EM in anom space (for plotting and for reconstructions later)
lens_em_u_adj_anom = lens_em_u_adj - np.mean(lens_em_u_adj.sel(time=slice(1961,1990)))
lens_em_psl_adj_anom = lens_em_psl_adj - np.mean(lens_em_psl_adj.sel(time=slice(1961,1990)))

#%%
#plot SLP to visualize anomaly ref period, and compare to ERA
plot_all_slp_data = True
if plot_all_slp_data:
    
    x,y = 0.95,0.92
    fig = plt.figure()
    fig.set_size_inches(12,12)

    fig.add_subplot(331)
    plt.plot(lens_em_time,lens_em_psl_adj,label='LENS historical EM adj\nmean 1979-2005=%.2f'%np.mean(lens_em_psl_adj[59:]),color='tab:blue',linestyle='--')
    #also plot ERA
    plt.plot(era_time,era_psl,color='k',label='ERA5\nmean=%.2f'%np.mean(era_psl))
    plt.legend(loc='upper left')
    plt.annotate('Mean = %.2f'%np.mean(lens_em_psl_adj)+\
                 '\n1961-1990 ref value = %.2f'%lens_em_psl_adj_ref,(0.05,0.05),\
                     xycoords='axes fraction')
    plt.annotate('a',(x,y),xycoords='axes fraction')
    
        
    fig.add_subplot(334)
    plt.plot(lens_em_time,lens_em_psl_adj_anom,label='LENS historical EM anom',color='tab:blue')
    plt.legend(loc='upper left')
    plt.annotate('Mean = %.2f'%np.mean(lens_em_psl_adj_anom),(0.05,0.05),\
                     xycoords='axes fraction')
    plt.annotate('b',(x,y),xycoords='axes fraction')
    
     
    fig.add_subplot(332)
    plt.plot(lens_pi_time,lens_pi_psl,label='LENS PI raw',color='gray',linestyle='--')
    plt.legend(loc='upper left')
    plt.annotate('Mean = %.2f'%np.mean(lens_pi_psl),(0.05,0.05),xycoords='axes fraction')
    plt.annotate('c',(x,y),xycoords='axes fraction')
    
    fig.add_subplot(335)
    plt.plot(lens_pi_time,lens_pi_psl_anom,label='LENS PI anomaly',color='gray')
    plt.legend(loc='upper left')
    plt.annotate('LENS PI - LENS EM ref\nMean = %.2f'%np.mean(lens_pi_psl_anom),(0.05,0.05),xycoords='axes fraction')
    plt.ylim([-12,15])
    plt.annotate('d',(x,y),xycoords='axes fraction')
    
    fig.add_subplot(333)
    plt.plot(lens_hist_time[0:86],lens_hist_psl[0:86],label='LENS hist E1 raw',color='lightsteelblue',linestyle='--')
    plt.legend(loc='upper left')
    plt.annotate('Mean = %.2f'%np.mean(lens_hist_psl[0:86]),(0.05,0.05),xycoords='axes fraction')
    plt.annotate('e',(x,y),xycoords='axes fraction')

    #plot LENS hist E1 as an example
    fig.add_subplot(336)
    lens_hist_e1_anom = lens_hist_psl[0:86]-np.tile(lens_em_psl_adj_ref,86)
    plt.plot(lens_hist_time[0:86],lens_hist_e1_anom,label='LENS hist E1 anomaly',color='lightsteelblue')
    plt.legend(loc='upper left')
    plt.annotate('LENS hist E1 - LENS EM ref\nMean = %.2f'%np.mean(lens_hist_e1_anom),(0.05,0.05),xycoords='axes fraction')
    plt.ylim([-10,8])
    plt.annotate('f',(x,y),xycoords='axes fraction')

    fig.add_subplot(339)
    # lens_hist_e1_anom_nat = lens_hist_e1_anom - lens_em_psl_anom
    lens_hist_e1_anom_nat = lens_hist_psl_adj_anom[0:86]
    plt.plot(lens_hist_time[0:86],lens_hist_e1_anom_nat,label='LENS hist E1 anom internal',color='navy')
    plt.legend(loc='upper left')
    plt.annotate('LENS hist E1 anom - LENS EM anom\nMean = %.2f'%np.mean(lens_hist_e1_anom_nat),(0.05,0.05),xycoords='axes fraction')
    plt.ylim([-10,8])
    plt.annotate('g',(x,y),xycoords='axes fraction')

    plt.suptitle('Model SLP')
    plt.subplots_adjust(left=0.05,right=.99,top=.95,bottom=.08,wspace=.14)
    
    #Plot U
    fig = plt.figure()
    fig.set_size_inches(12,12)

    fig.add_subplot(331)
    plt.plot(lens_em_time,lens_em_u_adj,label='LENS historical EM raw\nmean 1979-2005=%.2f'%np.mean(lens_em_u[59:]),color='tab:blue',linestyle='--')
    #also plot ERA
    plt.plot(era_time,era_u,color='k',label='ERA5\nmean=%.2f'%np.mean(era_u))
    plt.legend(loc='upper left')
    plt.annotate('Mean = %.2f'%np.mean(lens_em_u_adj)+\
                 '\n1961-1990 ref value = %.2f'%lens_em_u_adj_ref,(0.05,0.05),\
                     xycoords='axes fraction')
    plt.annotate('a',(x,y),xycoords='axes fraction')
    
    fig.add_subplot(334)
    plt.plot(lens_em_time,lens_em_u_adj_anom,label='LENS historical EM anom',color='tab:blue')
    plt.legend(loc='upper left')
    plt.annotate('Mean = %.2f'%np.mean(lens_em_u_adj_anom),(0.05,0.05),\
                     xycoords='axes fraction')
    plt.annotate('b',(x,y),xycoords='axes fraction')
    
    fig.add_subplot(332)
    plt.plot(lens_pi_time,lens_pi_u,label='LENS PI raw',color='gray',linestyle='--')
    plt.legend(loc='upper left')
    plt.annotate('Mean = %.2f'%np.mean(lens_pi_u),(0.05,0.05),xycoords='axes fraction')
    plt.annotate('c',(x,y),xycoords='axes fraction')
    
    fig.add_subplot(335)
    plt.plot(lens_pi_time,lens_pi_u_anom,label='LENS PI anomaly',color='gray')
    plt.legend(loc='upper left')
    plt.annotate('LENS PI - LENS EM ref\nMean = %.2f'%np.mean(lens_pi_u_anom),(0.05,0.05),xycoords='axes fraction')
    # plt.ylim([-12,15])
    plt.annotate('d',(x,y),xycoords='axes fraction')
    
    fig.add_subplot(333)
    plt.plot(lens_hist_time[0:86],lens_hist_u[0:86],label='LENS hist E1 raw',color='lightsteelblue',linestyle='--')
    plt.legend(loc='upper left')
    plt.annotate('Mean = %.2f'%np.mean(lens_hist_u[0:86]),(0.05,0.05),xycoords='axes fraction')
    plt.annotate('e',(x,y),xycoords='axes fraction')
    
    fig.add_subplot(336)
    lens_hist_e1_anom = lens_hist_u[0:86]-np.tile(lens_em_u_adj_ref,86)
    plt.plot(lens_hist_time[0:86],lens_hist_e1_anom,label='LENS hist E1 anomaly',color='lightsteelblue')
    plt.legend(loc='upper left')
    plt.annotate('LENS hist E1 - LENS EM ref\nMean = %.2f'%np.mean(lens_hist_e1_anom),(0.05,0.05),xycoords='axes fraction')
    # plt.ylim([-10,8])
    plt.annotate('f',(x,y),xycoords='axes fraction')
    
    fig.add_subplot(339)
    # lens_hist_e1_anom_nat = lens_hist_e1_anom - lens_em_u_anom
    lens_hist_e1_anom_nat = lens_hist_u_nat[0:86]
    plt.plot(lens_hist_time[0:86],lens_hist_e1_anom_nat,label='LENS hist E1 anom internal',color='navy')
    plt.legend(loc='upper left')
    plt.annotate('LENS hist E1 anom - LENS EM anom\nMean = %.2f'%np.mean(lens_hist_e1_anom_nat),(0.05,0.05),xycoords='axes fraction')
    # plt.ylim([-10,8])
    plt.annotate('g',(x,y),xycoords='axes fraction')
    
    plt.suptitle('Model Us')
    plt.subplots_adjust(left=0.05,right=.99,top=.95,bottom=.08,wspace=.14)

plot_model_data_simple = False
if plot_model_data_simple:
    fig=plt.figure()
    fig.add_subplot(311)
    plt.plot(lens_pi_time,lens_pi_u_anom,label='LENS PI anomaly',color='gray',linewidth=.7)
    plt.legend(loc='upper left')
    
    fig.add_subplot(312)
    plt.plot(lens_em_time,lens_em_u_adj,label='LENS Historical EM adj',\
             color='tab:blue',linewidth=.7)
    plt.legend(loc='lower left')
    
    fig.add_subplot(313)
    plt.plot(lens_hist_time,lens_hist_u_nat,label='LENS Hist - LENS EM adj',\
             color='lightsteelblue',linewidth=.7)
    plt.legend(loc='upper left')
    
    plt.suptitle('Model ASE shelf break U')
    plt.subplots_adjust(top=0.92)
    
    fig = plt.figure()
    fig.add_subplot(311)
    plt.plot(lens_pi_time,lens_pi_psl_anom,label='LENS PI anomaly',color='gray',linewidth=.7)
    plt.legend(loc='upper left')
    
    fig.add_subplot(312)
    plt.plot(lens_em_time,lens_em_psl_adj,label='LENS Historical EM adj',\
             color='tab:blue',linewidth=.7)
    plt.legend(loc='lower left')

    fig.add_subplot(313)
    plt.plot(lens_hist_time,lens_hist_psl_nat,label='LENS Hist - LENS EM adj',\
             color='lightsteelblue',linewidth=.7)
    plt.legend(loc='upper left')
    
    plt.suptitle('Model ASE shelf break SLP')
    plt.subplots_adjust(top=0.92)
    
#%% Get ensemble mean recon and isolate natural component

run72 = 'iCESM_LME_GKO1_all_lin_1mc_1900_2005_GISBrom_1880_2019_1deg_res_full_ens'
run73 = 'HadCM3_all_lin_1mc_1900_2005_GISBrom_1880_2019_1deg_res_full_ens'
run77 = 'LENS_super_GKO1_linPSM_1mc_1900_2005_GISBrom_1880_2019_1_deg_res_full_ens'
run78 = 'PACE_super_GKO1_all_linPSM_1mc_1900_2005_GISBrom_1880_2019_1_deg_res_full_ens'
run80 = 'LENS_super_GKO1_all_linPSM_1mc_1900_2005_GISBrom_1880_2019_v10'
run81 = 'LENS_preindustrial_GKO1_all_linPSM_1mc_1800_2005_GISBrom_1880_2019'
run82 = 'LENS_super_GKO1_all_linPSM_1mc_1800_2005_GISBrom_1880_2019'

cesm_run = run72
hadcm_run = run73
lens_run = run82
pace_run = run78

run = cesm_run
time_per = [1920,2005]

#get recon EM
recon_time, em_recon_u10 = load_1d_recon(run,'u10',time_per,'ASE')
recon_time, em_recon_psl = load_1d_recon(run,'psl',time_per,'ASE')

#remove LENS EM adjusted
em_recon_u10_nat = em_recon_u10 - lens_em_u_adj_anom
em_recon_psl_nat = em_recon_psl - lens_em_psl_adj_anom

#plot 
fig = plt.figure()
fig.add_subplot(211)
plt.plot(recon_time,em_recon_psl,label='Recon EM',color='tab:blue')
plt.plot(lens_em_time,lens_em_psl_adj_anom,label='LENS EM anomaly',color='gray')
plt.plot(recon_time,em_recon_psl_nat,label='Recon EM - LENS EM',color='navy')
plt.legend(ncol=3)
plt.ylabel('SLP anomaly (hPa')

fig.add_subplot(212)
plt.plot(recon_time,em_recon_u10,label='Recon EM',color='tab:blue')
plt.plot(lens_em_time,lens_em_u_adj_anom,label='LENS EM anomaly',color='gray')
plt.plot(recon_time,em_recon_u10_nat,label='Recon EM - LENS EM',color='navy')
plt.legend(ncol=3)
plt.ylabel('Us anomaly (m/s)')




#%% Generate full ensemble timeseries of reconstruction with natural component

#Use the same spread as the full recon ensemble (with the anthro component)

#Get the full ensemble of the reconstruction
recon_time,recon_u10_full_ens = load_1d_recon_full_ens(run,'u10',time_per,'ASE') #shape (86,100)
recon_time,recon_psl_full_ens = load_1d_recon_full_ens(run,'psl',time_per,'ASE')

def make_natural_full_ens(full_ens_data,em_data,nat_em_data):
    """
    Generate 100 ensemble members of the natural-component reconstruction 
    using the same spread as the full ensemble in the original reconstruction.
    Does this for the 1d regional data (i.e. ASE winds from 1920-2005).
    
    Method: calculates spacing of the 100 members relative to the recon EM
    for the original reconstruction based on the 1st year (since spread for each
    year is almost identical). Generates ensemble members centered around
    the natural recon using the same spacing. 

    Parameters
    ----------
    full_ens_data : 2d numpy array
        original 100 members of the recon. shape (n_yrs, n_ens)
    em_data : 1d numpy array
        original reconstruction ensemble mean. shape n_yrs
    nat_em_data : 1d numpy array
        reconstruction ensemble mean - LENS EM. shape n_yrs.

    Returns
    -------
    full_ens: 2d numpy array
        100 ensemble members centered around natural recon em. 
        shape 100,86 for n_ens,n_yrs

    """
    #get the spread from the full ensemble so you can add it to the natural timeseries
    #use the spacing based on the first year of data (since it's ~same for every year)
    yr1_full_ens = full_ens_data[0,:] #(100,)
    spacings = yr1_full_ens - em_data[0] #get differences between em and each member.shape 100
    
    #construct members by adding natural em for each year to each spacing
    n_ens = len(spacings) #100
    n_yrs = len(em_data)
    full_ens = []
    for i in range(n_ens):
        ens_i_data = []
        ens_spacing_value = spacings[i]
        for j in range(n_yrs):
            ens_i_data.append(nat_em_data[j] + ens_spacing_value)
        full_ens.append(ens_i_data)
    full_ens = np.array(full_ens) #shape 100, 86
    
    return full_ens
        

nat_full_ens_u = make_natural_full_ens(recon_u10_full_ens,em_recon_u10,\
                                       em_recon_u10_nat)
nat_full_ens_psl = make_natural_full_ens(recon_psl_full_ens,em_recon_psl,\
                                       em_recon_psl_nat)
    
    
#plot natural full ensemble
fig = plt.figure()
for i in range(100):
    # plt.plot(recon_time,nat_full_ens_u[i])
    plt.plot(recon_time,nat_full_ens_psl[i])
    plt.title('Full ensemble of natural component of reconstruction')
    
#plot a randomly drawn recon ensemble member vs models to compare variability
fig = plt.figure()

fig.add_subplot(211)
plt.plot(recon_time,nat_full_ens_psl[0,:],label = '1st scrambled recon member (internal)',color='k')
plt.plot(lens_pi_time,lens_pi_psl_anom,label='LENS PI anomaly',color='gray')
plt.plot(lens_hist_time,lens_hist_psl_nat,label='LENS hist - LENS EM',color='tab:blue')
plt.legend(ncol=2,loc='upper left')
plt.xlim([1920,2005])
plt.ylabel('SLP anomaly (hPa)')

fig.add_subplot(212)
plt.plot(recon_time,nat_full_ens_u[0,:],label = '1st scrambled recon member (internal)',color='k')
plt.plot(lens_pi_time,lens_pi_u_anom,label='LENS PI anomaly',color='gray')
plt.plot(lens_hist_time,lens_hist_u_nat,label='LENS hist - LENS EM',color='tab:blue')
# plt.legend()
plt.ylabel('U anomaly (m/s)')
plt.xlim([1920,2005])


#%% Scramble natural full ensemble of recon

def calc_scram_ens(full_ens_data,time,n_draws):
    """
    Calculates n_ens timeseries by randomly drawing from recon ensemble members.

    Parameters
    ----------
    full_ens_data : 2d numpy array
        recon full ensemble data (probably shape 100,86)
    time : 1d numpy array
        years associated with full ens data
    n_draws : int
        how many timeseries of scrambled ensemble members to make

    Returns
    -------
    scram_ens : 2d numpy array
        shape n_draws, n_years

    """
    
    n_yrs = len(time)
    scram_ens = np.zeros((n_draws,n_yrs))
    #iterate through number of random timeseries you want
    for i in range(n_draws):
    
        # print('working on random draw',i)
        rand_timeseries_i = []
        #get random draw for each year in timeseries
        for year_idx in range(len(time)):
            rand_idx = int(np.random.uniform(low=0,high=99))#inclusive
            rand_val = full_ens_data[rand_idx,year_idx]
            rand_timeseries_i.append(rand_val)
        rand_timeseries_i = np.array(rand_timeseries_i) #has shape n_years,n_lats,n_lons

        scram_ens[i,:] = rand_timeseries_i
    
    return scram_ens

n_draws = 200

scram_ens_u = calc_scram_ens(nat_full_ens_u,recon_time,n_draws) #shape n_draws,86
scram_ens_psl = calc_scram_ens(nat_full_ens_psl,recon_time,n_draws)

#plot scrambled ensemble members
fig = plt.figure()
for draw in range(n_draws):
    plt.plot(recon_time,scram_ens_u[draw,:])
    # plt.plot(recon_time,scram_ens_psl[draw,:])
plt.title(str(n_draws)+' randomly drawn timeseries from full ensemble')
plt.ylabel('Us (m/s)')


#%% Calculate thresholds


#For each scrambled tseries, Calculate the mean value maintained for m years

#order for 1,2,3...,10-yr long events centered around 1940 (1940/41 is biggest)
start_yrs = [1940,1940,1939,1939,1938,1938,1937,1937,1936,1936]
stop_yrs = [1940,1941,1941,1942,1942,1943,1943,1944,1944,1945] #use indexes of yrs + 1


def calc_thresholds(years,recon_data,start_yrs,stop_yrs):
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
        list of mean values maintained in the mean of all 200 recon members
        for each m-yr event. order follows order of events corresponding to
        start_yrs (i.e. 1,2,3...,10-yr events)
    low_thresholds : list
        lower bound of 95% confidence interval threshold for all events
    upp_thresholds : list
        upper bound of 95% confidence interval threshold for all events

    """
    n_events = len(start_yrs)
    
    #for each event type
    mean_thresholds = [] #len 10
    low_thresholds = []
    upp_thresholds = []
    for i in range(n_events):
        
        #m_yr events are 1,2,...,10
        m_yrs = i + 1
        print('looking at '+str(m_yrs)+'-yr events...')
        
        #get start index and stop index for this event, centered around 1940
        start_idx = np.where(years == start_yrs[i])[0][0]
        stop_idx = np.where(years == stop_yrs[i]+1)[0][0]
        
        #thresholds for an m-yr event based on the 200 tseries 
        scram_thresholds = [] #len 200
        
        #for each scrambled tseries (t goes through each of 200 tseries)
        for tseries in recon_data:
            
            #get values during event
            tseries_event = tseries[start_idx:stop_idx]
            if len(scram_thresholds) == 0:
                print(years[start_idx:stop_idx])
            event_mean = np.mean(tseries_event)
            scram_thresholds.append(event_mean)
        
    
        #calculate mean and 95% CI of scram thresholds
        mean = np.mean(scram_thresholds)
        #use normal distribution bc sample size > 30
        low_ci,upp_ci = stats.norm.interval(alpha=0.95, loc=mean, \
                                         scale=stats.sem(scram_thresholds))
        print('Mean = %.2f, ' %mean + '95% CI = ['+'%.2f,' %low_ci + '%.2f]\n' %upp_ci)
        
        mean_thresholds.append(mean)
        low_thresholds.append(low_ci)
        upp_thresholds.append(upp_ci)
        
    return mean_thresholds,low_thresholds,upp_thresholds
        
mean_u_thresh,low_u_thresh,upp_u_thresh = calc_thresholds(recon_time,\
                scram_ens_u,start_yrs,stop_yrs)
mean_psl_thresh,low_psl_thresh,upp_psl_thresh = calc_thresholds(recon_time,\
                scram_ens_psl,start_yrs,stop_yrs)



#%% Search model for occurrences exceeding thresholds

#Using an m-year sliding window so you are looking at n_model_years - m possible m-yr events
#(I.e. if you have 1000 years, you search 999 2-year events)

def search_model(model_data,events,thresholds):
    """
    For a given set of n thresholds corresponding to n m-yr long events, 
    search a model time series for m-year-long events meeting threshold.

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

    """
    
    #will make list of len n_events
    all_event_counters = []
    
    #search for each type of event
    for i in range(len(events)):
        
        m_yrs = int(events[i])
        # print('Searching ',m_yrs,'-yr events...')
        
        event_counter = 0
        event_thresh = thresholds[i]
        
        #search in each sliding window possible in the model
        for j in range(len(model_data)):
            # print('j:',j)
            
            event_j = model_data[j:j+m_yrs]
            # print('Checking event from: ',lens_pi_time[j:j+m_yrs])
            
            event_j_mean = np.mean(event_j)
            if event_j_mean >= event_thresh:
                event_counter += 1
        
        # print(event_counter,' events found!')
        all_event_counters.append(event_counter)
    
    return all_event_counters


events = np.linspace(1,10,10)

#SLP occurrences-----------------------------------

#Check LENS PI
print('Checking LENS PI for SLP events...')
psl_events_lens_pi_mean = search_model(lens_pi_psl_anom,events,mean_psl_thresh)
print('Number of events found using mean threshold:',psl_events_lens_pi_mean)
psl_events_lens_pi_low = search_model(lens_pi_psl_anom,events,low_psl_thresh)
print('Number of events found using lower threshold:',psl_events_lens_pi_low)
psl_events_lens_pi_upp = search_model(lens_pi_psl_anom,events,upp_psl_thresh)
print('Number of events found using upper threshold:',psl_events_lens_pi_upp)

#LENS hist internal component
print('Checking LENS historical natural component for SLP events...')
psl_events_lens_hist_nat_mean = search_model(lens_hist_psl_nat,events,mean_psl_thresh)
print('Number of events found using mean threshold:',psl_events_lens_hist_nat_mean)
psl_events_lens_hist_nat_low = search_model(lens_hist_psl_nat,events,low_psl_thresh)
print('Number of events found using lower threshold:',psl_events_lens_hist_nat_low)
psl_events_lens_hist_nat_upp = search_model(lens_hist_psl_nat,events,upp_psl_thresh)
print('Number of events found using upper threshold:',psl_events_lens_hist_nat_upp)

#LENS Hist al components
print('Checking LENS historical (all components) for SLP events...')
psl_events_lens_hist_mean = search_model(lens_hist_psl_adj_anom,events,mean_psl_thresh)
print('Number of events found using mean threshold:',psl_events_lens_hist_mean)
psl_events_lens_hist_low = search_model(lens_hist_psl_adj_anom,events,low_psl_thresh)
print('Number of events found using lower threshold:',psl_events_lens_hist_low)
psl_events_lens_hist_upp = search_model(lens_hist_psl_adj_anom,events,upp_psl_thresh)
print('Number of events found using upper threshold:',psl_events_lens_hist_upp)

#Check U occurrences---------------------------

#LENS PI
print('Checking LENS PI for wind events...')
u_events_lens_pi_mean = search_model(lens_pi_u_anom,events,mean_u_thresh)
print('Number of events found using mean threshold:',u_events_lens_pi_mean)
u_events_lens_pi_low = search_model(lens_pi_u_anom,events,low_u_thresh)
print('Number of events found using lower threshold:',u_events_lens_pi_low)
u_events_lens_pi_upp = search_model(lens_pi_u_anom,events,upp_u_thresh)
print('Number of events found using upper threshold:',u_events_lens_pi_upp)

#Check LENS historical natural component
print('Checking LENS historical natural component for wind events...')
u_events_lens_hist_nat_mean = search_model(lens_hist_u_nat,events,mean_u_thresh)
print('Number of events found using mean threshold:',u_events_lens_hist_nat_mean)
u_events_lens_hist_nat_low = search_model(lens_hist_u_nat,events,low_u_thresh)
print('Number of events found using lower threshold:',u_events_lens_hist_nat_low)
u_events_lens_hist_nat_upp = search_model(lens_hist_u_nat,events,upp_u_thresh)
print('Number of events found using upper threshold:',u_events_lens_hist_nat_upp)

#Check LENS historical (with anthropogenic component)
print('Checking LENS historical (all components) for wind events...')
u_events_lens_hist_mean = search_model(lens_hist_u_adj_anom,events,mean_u_thresh)
print('Number of events found using mean threshold:',u_events_lens_hist_mean)
u_events_lens_hist_low = search_model(lens_hist_u_adj_anom,events,low_u_thresh)
print('Number of events found using lower threshold:',u_events_lens_hist_low)
u_events_lens_hist_upp = search_model(lens_hist_u_adj_anom,events,upp_u_thresh)
print('Number of events found using upper threshold:',u_events_lens_hist_upp)


# print('Total U occurrences using mean threshold (n=1801+3440 years = 5241 years:')
# for i,j,k in zip(events,u_events_lens_pi_mean,u_events_lens_hist_nat_mean):
    
#     print(str(i)+'-year events:')
#     total_occurrences = j + k
#     prob = total_occurrences / 1801+3241
#     print(prob)

#%% 
#plot thresholds and model occurrences


fig = plt.figure()
fig.set_size_inches(10,8)

fig.add_subplot(221)
plt.grid(alpha=.5)
plt.scatter(events,mean_psl_thresh,label='Mean')
plt.scatter(events,low_psl_thresh,label='95% CI Lower Bound')
plt.scatter(events,upp_psl_thresh,label='95% CI Upper Bound')
plt.ylabel('Threshold')
plt.title('SLP Thresholds')
plt.xticks(events)

fig.add_subplot(222)
plt.grid(alpha=.5)
plt.scatter(events,mean_u_thresh,label='Mean')
plt.scatter(events,low_u_thresh,label='95% CI Lower Bound')
plt.scatter(events,upp_u_thresh,label='95% CI Upper Bound')
plt.legend()
plt.title('U Thresholds')
plt.ylabel('Threshold')
plt.xticks(events)

#plot SLP occurences 
fig.add_subplot(223)
space = 0.3
width = .25
plt.grid(alpha=.5)
#plot PI occurrences
plt.bar(events,psl_events_lens_pi_low,label='low thresh (LENS PI)',width=width,color='lightgreen')
plt.bar(events,psl_events_lens_pi_mean,label='mean thresh (LENS PI)',width=width,color='tab:green')
plt.bar(events,psl_events_lens_pi_upp,label='high thresh (LENS PI)',width=width,color='darkgreen')
#plot LENS hist internal component occurrences
plt.bar(events+space,psl_events_lens_hist_nat_low,label='low thresh (LH internal)',width=width,color='lightsteelblue')
plt.bar(events+space,psl_events_lens_hist_nat_mean,label='mean thresh (LH internal)',width=width,color='tab:blue')
plt.bar(events+space,psl_events_lens_hist_nat_upp,label='high thresh (LH internal)',width=width,color='navy')
#plot LENS historical all components occurrences
plt.bar(events+2*space,psl_events_lens_hist_low,label='low thresh (LH all)',width=width,color='gold')
plt.bar(events+2*space,psl_events_lens_hist_mean,label='mean thresh (LH all)',width=width,color='tab:orange')
plt.bar(events+2*space,psl_events_lens_hist_upp,label='high thresh (LH all)',width=width,color='maroon')
# plt.legend(fontsize=9)
plt.ylabel('Occurrences')
plt.title('SLP Occurrences')
plt.xlabel('Number of years in event')
plt.xticks(events)

#plot U occurrences
fig.add_subplot(224)
#plot LENS PI occurrences
plt.grid(alpha=.5)
plt.bar(events,u_events_lens_pi_low,label='low thresh (LENS PI)',width=width,color='lightgreen')
plt.bar(events,u_events_lens_pi_mean,label='mean thresh (LENS PI)',width=width,color='tab:green')
plt.bar(events,u_events_lens_pi_upp,label='high thresh (LENS PI)',width=width,color='darkgreen')
#plot LENS hist internal
plt.bar(events+space,u_events_lens_hist_nat_low,label='low thresh (LH internal)',width=width,color='lightsteelblue')
plt.bar(events+space,u_events_lens_hist_nat_mean,label='mean thresh (LH internal)',width=width,color='tab:blue')
plt.bar(events+space,u_events_lens_hist_nat_upp,label='high thresh (LH internal)',width=width,color='navy')
#plot hist with anthro occurrences

plt.bar(events+2*space,u_events_lens_hist_low,label='low thresh (LH all)',width=width,color='gold')
plt.bar(events+2*space,u_events_lens_hist_mean,label='mean thresh (LH all)',width=width,color='tab:orange')
plt.bar(events+2*space,u_events_lens_hist_upp,label='high thresh (LH all)',width=width,color='maroon')
plt.legend(fontsize=9)
plt.ylabel('Occurrences')
plt.title('U Occurrences')
plt.xlabel('Number of years in event')
plt.xticks(events)

plt.subplots_adjust(hspace=.35,top=.95,left=.1,right=.95)








