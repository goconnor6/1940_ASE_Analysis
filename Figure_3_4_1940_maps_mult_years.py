#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:57:13 2021

make maps of TAS, SLP, and US anomalies for specified years 
in two reconstructions.

Used to create Figures 2, 3, and S2 in O'Connor et al. 1940 analysis

@author: gemma
"""



import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from Functions_1940_analysis import load_3d_data, region_dict
from Datasets_1940_Analysis import *




start,stop = 1937,1943  

# For Figure 2
recon1, recon2 = cesm_recon, pace_recon

# For Figure 3
# recon1, recon2 = cesm_ice_recon, cesm_coral_recon

# For Figure S2
# recon1, recon2 = pace_ice_recon, pace_coral_recon 

#%% Get data
        
time_per = [start,stop]

tas_r1 = load_3d_data(recon1.path.replace('*vname*','tas'),'tas',time_per)
psl_r1 = load_3d_data(recon1.path.replace('*vname*','psl'),'psl',time_per)
u10_r1 = load_3d_data(recon1.path.replace('*vname*','u10'),'u10',time_per)
    
tas_r2 = load_3d_data(recon2.path.replace('*vname*','tas'),'tas',time_per)
psl_r2 = load_3d_data(recon2.path.replace('*vname*','psl'),'psl',time_per)
u10_r2 = load_3d_data(recon2.path.replace('*vname*','u10'),'u10',time_per)

#%% Get bathymetry data to plot shelf break

ds = xr.open_dataset('Data/IBCSO_v2_bed.nc')

    
#%% Plot data

datasets = [tas_r1, psl_r1, u10_r1, tas_r2, psl_r2, u10_r2]
lon, lat = tas_r1.lon, tas_r1.lat

years = np.linspace(start,stop,stop-start+1,dtype=int)
n_years = len(years)
n_rows = len(years)
n_cols = 6
fs = 10

# Set up colorbar lims for different vars
tas_lim = 2
p_lim = 3.5
u_lim = 1.5
n_levs = 9

# Plot one year at a time 
maps = []
fig = plt.figure()
fig.set_size_inches(6.5,5)
for col in range(n_cols):
    
    for year_idx in range(len(years)):
        
        fig_num = col+ 6*year_idx + 1
        ax = fig.add_subplot(n_rows,n_cols,fig_num)
        
        #Set up map based on variable type
        
        # Temp
        if col%3 == 0:
            mp = Basemap(projection='gall',llcrnrlat=-83,urcrnrlat=25,\
                          llcrnrlon=135,urcrnrlon=300,resolution='c')
            n_lat_lines = 30
            levs = np.linspace(-tas_lim,tas_lim,n_levs)
            cm = 'PiYG_r'
            if col == 0:
                plt.ylabel(str(int(years[year_idx])),fontsize=fs,labelpad=16)
        
        # SLP
        elif col%3 == 1:
            # mp = Basemap(projection='lcc',lat_0=-65,lon_0=250,\
            #               resolution='c',width=6e6,height=4e6)
            mp = Basemap(projection='lcc',lat_0=-60,lon_0=250,\
                          resolution='c',width=8.5e6,height=5.5e6)
            n_lat_lines = 20
            levs = np.linspace(-p_lim,p_lim,n_levs)
            cm = 'RdBu_r'
            if year_idx == 0:
                if col == 1:
                    plt.title('a) '+recon1.name + 'struction',pad=5,fontsize=fs-1)
                else:
                    plt.title('b) '+recon2.name + 'struction',pad=5,fontsize=fs-1)
        
        # U10
        elif col%3 == 2:
            mp = Basemap(projection='lcc',lat_0=-71,lon_0=250,\
                          resolution='c',width=3.5e6,height=2.3e6)
            # mp = Basemap(projection='lcc',lat_0=-65,lon_0=250,\
            #               resolution='c',width=6e6,height=4e6)
            n_lat_lines = 10
            levs = np.linspace(-u_lim,u_lim,n_levs)
            cm = 'PuOr_r'
            
            # Add box in ASE shelf break region
            lat1,lat2,lon1,lon2 = region_dict['ASE SB']
            x1,y1 = mp(lon2,lat1) #lower right lon (neg lon means west),lower right lat
            x2,y2 = mp(lon2,lat2) #upper right lon, lat
            x3,y3 = mp(lon1,lat2) #upper left
            x4,y4 = mp(lon1,lat1)
            poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],fill=False, \
                                edgecolor='black',linewidth=1)
            plt.gca().add_patch(poly)
            
        mp.drawparallels(np.arange(-90,90,n_lat_lines),\
                              labels=[True,False,False,False],\
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
        if col%3 == 2:
            plt.contour()
        
    maps.append(mp)

plt.subplots_adjust(left = 0.06,right = 0.99, bottom = 0.1, top = 0.95, wspace = 0.25)

# Add colorbars
xpos = .068
width = .12
height = 0.017
ypos = [.064,.225,.385,.547,.707,.866]
labs = ['TAS (C)','SLP (hPa)',r'U$_S$ (m/s)'] * 2
lims = [tas_lim,p_lim,u_lim] * 2

for col in range(n_cols):
    cb_ax = fig.add_axes([ypos[col],xpos,width,height])
    cb = fig.colorbar(maps[col], cax=cb_ax, extend='both',orientation = 'horizontal')
    cb.set_label(label = labs[col], fontsize=fs-2,labelpad=0)
    cb.ax.tick_params(labelsize=fs-3)
    cb.set_ticks([-lims[col],0,lims[col]])
    cb.set_ticklabels([-lims[col],0,lims[col]])


# Uncomment to save fig
# plt.savefig('Plots/Figure_2_1940_maps_all_proxies.png', bbox_inches = 'tight',dpi=600) 
# plt.savefig('Plots/Figure_3_1940_maps_cesm_single_proxies.png', bbox_inches = 'tight',dpi=600) 
# plt.savefig('Plots/Figure_S2_1940_maps_pace_single_proxies.png', bbox_inches = 'tight',dpi=600) 

