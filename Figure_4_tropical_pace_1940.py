#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:10:12 2022

Plot PACE simulation ensemble mean and 2 selected members for 5 years
Makes figure 4 in O'Connor et al. 1940 paper

@author: gemma
"""


import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from Functions_1940_analysis import load_3d_data, region_dict
from Datasets_1940_Analysis import *

#%% 

# Specify which two ensemble members from the Pacific Pacemaker ensemble to plot
members = ['07','18']

# Which 5 years to plot (>5 ok, but figure was designed for 5 rows)
time_per = [1938,1942] #Inclusive

anom_ref = [1961,1990]
region = 'South Pacific' #refers to which data to query, not the plotting area

# Option to plot all members from the Pacemaker simulation (helpful for selecting 2 members)
show_all_members = False

#%% Load PACE model ensemble mean


#Get PACE EM in anomaly space
pace_em_psl_anom = load_3d_data(pac_pace_model.em_path.replace('*vname*','psl'),\
                           'psl', time_per, region=region, \
                           anom_ref = anom_ref)
pace_em_u10_anom = load_3d_data(pac_pace_model.em_path.replace('*vname*','u10'),\
                           'u10', time_per, region=region, \
                           anom_ref = anom_ref)

#%% Get specified pace members in anomaly space relative to EM
    
# First get PACE EM ref values for all locs:
pace_em_psl_ref_data = load_3d_data(pac_pace_model.em_path.replace('*vname*','psl'),\
                           'psl', anom_ref, region='South Pacific')
pace_em_psl_ref = pace_em_psl_ref_data.mean(dim='time')
pace_em_u10_ref_data = load_3d_data(pac_pace_model.em_path.replace('*vname*','u10'),\
                           'u10', anom_ref, region='South Pacific')
pace_em_u10_ref = pace_em_u10_ref_data.mean(dim='time')


# Then remove the EM ref vals from each member:
mem_psl_data = []
mem_u10_data = []
for member in members:
    
    #SLP
    member_psl_anom = pac_pace_model.remove_model_em_ref(member,'psl',\
                      time_per,anom_ref,region=region)
    mem_psl_data.append(member_psl_anom)
    
    #U10
    member_u10_anom = pac_pace_model.remove_model_em_ref(member,'u10',\
                      time_per,anom_ref,region=region)
    mem_u10_data.append(member_u10_anom)




#%% Plot PACE EM and 2 selected members (both SLP and US)

#want to plot from left to right, top to bottom
#PACE EM SLP, PACE E7 SLP, PACE E18 SLP, then repeat for wind
#top row is 1938, bottom row 1942


datasets = [pace_em_psl_anom, mem_psl_data[0],mem_psl_data[1],\
             pace_em_u10_anom, mem_u10_data[0],mem_u10_data[1]]
lon, lat = pace_em_psl_anom.lon, pace_em_psl_anom.lat

labels = ['PACE EM',f'PACE Ens {int(mems[0])}',f'PACE Ens {int(mems[1])}']*2
letters = ['a','b','c','d','e','f']
start,stop = time_per
years = np.linspace(start,stop,stop-start+1,dtype=int)
n_years = len(years)
n_cols = len(datasets)
fs = 10

# Set up colorbar lims for different vars
tas_lim = 2
p_lim = 3.5
u_lim = 1.5
n_levs = 9

# Plot one year at a time 
maps = []
fig = plt.figure()
fig.set_size_inches(5.7,3.5)
for col in range(n_cols):
    
    for year_idx in range(n_years):
        
        fig_num = col+ 6*year_idx + 1
        ax = fig.add_subplot(n_years,n_cols,fig_num)
        
        # SLP
        if col < 3:
            mp = Basemap(projection='lcc',lat_0=-60,lon_0=250,\
                          resolution='c',width=8.5e6,height=5.5e6)
            n_lat_lines = 20
            levs = np.linspace(-p_lim,p_lim,n_levs)
            cm = 'RdBu_r'
            if year_idx == 0:
                plt.title(letters[col]+') '+labels[col],pad=5,fontsize=fs-2)
            if col == 0:
                plt.ylabel(years[year_idx],fontsize=fs-1,labelpad=16)
                mp.drawparallels(np.arange(-90,90,n_lat_lines),\
                              labels=[True,False,False,False],\
                              fontsize=fs-4,linewidth=0.5)
        # U10
        else:
            mp = Basemap(projection='lcc',lat_0=-71,lon_0=250,\
                          resolution='c',width=3.5e6,height=2.3e6)
            n_lat_lines = 10
            levs = np.linspace(-u_lim,u_lim,n_levs)
            cm = 'PuOr_r'
            if year_idx == 0:
                plt.title(letters[col]+') '+labels[col],pad=5,fontsize=fs-2)
            if col == n_cols-1:
                mp.drawparallels(np.arange(-90,90,n_lat_lines),\
                              labels=[False,True,False,False],\
                              fontsize=fs-4,linewidth=0.5)
            # Add box in ASE shelf break region
            lat1,lat2,lon1,lon2 = region_dict['ASE SB']
            x1,y1 = mp(lon2,lat1) 
            x2,y2 = mp(lon2,lat2) 
            x3,y3 = mp(lon1,lat2) 
            x4,y4 = mp(lon1,lat1)
            poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],fill=False, \
                                edgecolor='black',linewidth=1)
            plt.gca().add_patch(poly)
            
        mp.drawparallels(np.arange(-90,90,n_lat_lines),\
                              labels=[False,False,False,False],\
                              fontsize=fs-4,linewidth=0.5)
        mp.drawmeridians(np.arange(0,360,50),
                          labels=[False,False,False,False],\
                          linewidth=0.5)
        mp.drawcoastlines()
        mp.drawmapboundary()
        mp.fillcontinents()
        
        # Plot data
        x,y = mp(*np.meshgrid(lon,lat))
        data = datasets[col].sel(time = years[year_idx])
        mp=mp.contourf(x,y,data,cmap=cm,extend='both',levels=levs) 
        
    maps.append(mp)

plt.subplots_adjust(left = 0.06,right = 0.95, bottom = 0.2, top = 0.96, wspace = 0.25)

# Add 2 colorbars
xpos = .09
width = .18
height = 0.017
ypos = [.2,.65,]
labs = ['SLP (hPa)',r'U$_S$ (m/s)'] 
lims = [p_lim,u_lim] 
cb_maps = [maps[0],maps[3]]

for i in range(2):
    cb_ax = fig.add_axes([ypos[i],xpos,width,height])
    cb = fig.colorbar(cb_maps[i], cax=cb_ax, extend='both',orientation = 'horizontal')
    cb.set_label(label = labs[i], fontsize=fs-2,labelpad=0)
    cb.ax.tick_params(labelsize=fs-3)
    cb.set_ticks([-lims[i],0,lims[i]])
    cb.set_ticklabels([-lims[i],0,lims[i]])
        
plt.subplots_adjust(wspace=0.1,hspace=0.1,top = 0.94,bottom=0.12,right=0.96,left=0.07)

# plt.savefig('Plots/Figure_4_tropical_pacemaker_1940_maps.png',dpi=600)

#%% Plot all ensemble members for a given year

def plot_all_members_one_year(year,data,vname):
        """
        Plot 20 Pacemaker ensemble members for SLP or U10
        Makes a 4 x 5 plot

        Parameters
        ----------
        year : int of year
            Only used for title
        data : np array
            data for specified year with shape (n_lats, n_lons, n_mems)
        vname : str
            either psl or u10

        Returns
        -------
        None. (Displays map)

        """

        n_cols = 4
        n_rows = 5
        n_ens = data.shape[-1]
        fig = plt.figure()
        fig.set_size_inches(4,5)
        fig_num = 1
        
        while fig_num <= n_ens:
            
            fig.add_subplot(n_rows,n_cols,fig_num)
            
            # SLP
            if vname == 'psl':
                mp = Basemap(projection='lcc',lat_0=-60,lon_0=250,\
                              resolution='c',width=8.5e6,height=5.5e6)
                n_lat_lines = 20
                levs = np.linspace(-p_lim,p_lim,n_levs)
                cm = 'RdBu_r'
                plt.title(f'Ensemble {fig_num}',fontsize=fs-2)

            # U10
            elif vname == 'u10':
                mp = Basemap(projection='lcc',lat_0=-71,lon_0=250,\
                              resolution='c',width=3.5e6,height=2.3e6)
                n_lat_lines = 10
                levs = np.linspace(-u_lim,u_lim,n_levs)
                cm = 'PuOr_r'
                if col == n_cols-1:
                    mp.drawparallels(np.arange(-90,90,n_lat_lines),\
                                  labels=[False,True,False,False],\
                                  fontsize=fs-4,linewidth=0.5)
                # Add box in ASE shelf break region
                lat1,lat2,lon1,lon2 = region_dict['ASE SB']
                x1,y1 = mp(lon2,lat1) 
                x2,y2 = mp(lon2,lat2) 
                x3,y3 = mp(lon1,lat2) 
                x4,y4 = mp(lon1,lat1)
                poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],fill=False, \
                                    edgecolor='black',linewidth=1)
                plt.gca().add_patch(poly)
                
            mp.drawparallels(np.arange(-90,90,n_lat_lines),\
                                  labels=[False,False,False,False],\
                                  fontsize=fs-4,linewidth=0.5)
            mp.drawmeridians(np.arange(0,360,50),
                              labels=[False,False,False,False],\
                              linewidth=0.5)
            mp.drawcoastlines()
            mp.drawmapboundary()
            mp.fillcontinents()
            
            # Plot data
            x,y = mp(*np.meshgrid(lon,lat))
            mem_data = data[:,:,int(fig_num)-1]
            mp=mp.contourf(x,y,mem_data,cmap=cm,extend='both',levels=levs) 
            
            fig_num += 1
        
        plt.suptitle(str(year))
        plt.subplots_adjust(left = 0.06,right = 0.95, bottom = 0.2, top = 0.96, wspace = 0.25)
        
        # Add colorbar
        if vname == 'psl':
            lim = 3.5
        elif vname == 'u10':
            lim = 1.5
        cb_ax = fig.add_axes([.35,.09,.3,.017]) # xpos, ypos, width, height
        cb = fig.colorbar(mp, cax=cb_ax, extend='both',orientation = 'horizontal')
        cb.set_label(label = vname, fontsize=fs-2,labelpad=0)
        cb.ax.tick_params(labelsize=fs-3)
        cb.set_ticks([-lim,0,lim])
        cb.set_ticklabels([-lim,0,lim])
                
        plt.subplots_adjust(wspace=0.1,hspace=0.15,top = 0.9,bottom=0.12,right=0.96,left=0.07)
        
        return
    
if show_all_members:
    
    # Get data for all ensemble members
    all_members = pac_pace_model.members
    n_mems = len(all_members)
    n_lats, n_lons = len(lat), len(lon)
    mem_psl_data = np.zeros((n_years, n_lats, n_lons, n_mems)) #these are np arrays
    mem_u10_data = np.zeros((n_years, n_lats, n_lons, n_mems))
    
    for i in range(n_mems):
        
        #SLP
        member_psl_anom = pac_pace_model.remove_model_em_ref(all_members[i],'psl',\
                          time_per,anom_ref,region=region)
        mem_psl_data[:,:,:,i] = member_psl_anom
        
        #U10
        member_u10_anom = pac_pace_model.remove_model_em_ref(all_members[i],'u10',\
                          time_per,anom_ref,region=region)
        mem_u10_data[:,:,:,i] = member_u10_anom
    


    # Map all ensemble members for a given year (plug and play with year)
    year = 1940
    
    year_idx = np.where(years == year)[0][0]
    plot_all_members_one_year(year,mem_psl_data[year_idx,:,:,:],'psl')
    plot_all_members_one_year(year,mem_u10_data[year_idx,:,:,:],'u10')
