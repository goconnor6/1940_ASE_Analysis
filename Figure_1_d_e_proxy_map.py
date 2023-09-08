#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:36:17 2023

plot proxy data on map with ERA5 mean SLP and US fields in color.
as additional panels to figure 1 in 1940s paper.

@author: gemma
"""

import pandas as pd
import numpy as np
import pickle
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import xarray as xr
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely.geometry import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable



#%% Get proxy locations

#reading in pckl file with sparse dataframe type
df_list = []
with open('../Proxies/LMRdb_vGKO1_Metadata.df.pckl', 'rb') as pickle_file:
    try:
        while True:
            test =  pickle.load(pickle_file)
            df_list.append(test)
    except EOFError:
        pass
df_2 = pd.concat((df_list), ignore_index= True)
meta = np.array(df_2)

#finding corals and ice cores
ice_indices = np.where(meta == 'Ice Cores') #190 ice cores total (correct)
coral_indices = np.where(meta == 'Corals and Sclerosponges') #139 corals total (correct)
tree_indices = np.where(meta == 'Tree Rings')
proxy_indices = np.concatenate((ice_indices[0],coral_indices[0],tree_indices[0]))
color_dict = {'Tree Rings':'#006d2c','Corals and Sclerosponges':'magenta',\
              'Ice Cores':'turquoise'}
proxy_dict = {}
tree_dict = {}
ice_dict = {}
coral_dict = {}
for i in range(len(proxy_indices)):
    info = meta[proxy_indices[i]] 
    name = info[0]
    lat = info[4]
    lon = info[5]
    archive = info[7]
    color = color_dict[archive]
    if 'breit' not in info[8]:
        proxy_dict[name] = [lat,lon,color]
    if archive == 'Tree Rings':
        tree_dict[name] = [lat,lon]
    elif archive == 'Corals and Sclerosponges':
        coral_dict[name] = [lat,lon]
    elif archive == 'Ice Cores':
        ice_dict[name] = [lat,lon]
        
#%% Get ERA5 mean slp and US fields

ds = xr.open_dataset('Data/Verification/ERA5/annual_psl_1979_2019.nc')
psl = ds.psl
psl_mean = psl.mean(axis=0)
psl_mean = psl_mean/100

ds = xr.open_dataset('Data/Verification/ERA5/annual_u1000_1979_2019.nc')
u10 = ds.u1000
u10_mean = u10.mean(axis=0)


#%% make map

plot_data = [psl_mean, u10_mean]
labels = ['SLP (hPa)','U$_S$ (m/s)']
lims = [[980,1020],[-8,8]]
cmaps = ['RdBu_r','PuOr_r']

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(3, 4.5), \
                         subplot_kw={'projection': ccrs.Orthographic(central_longitude = -130,\
                                                                    central_latitude = -45)})

for i in range(2):

    ax = axes[i]
    data = plot_data[i]
    low,high = lims[i]
    # contourf cuts off edge when plotting proxies...spent too much time looking into this
    mp = ax.contourf(data.lon, data.lat, data, transform=ccrs.PlateCarree(), \
                          cmap=cmaps[i],levels=np.linspace(low,high,11),extend='both')
    
    # Add colorbar
    cbar=plt.colorbar(mp,pad=.08,extend='both',shrink=0.7)
    cbar.set_label(label = labels[i], fontsize=9)
    cbar.set_ticks(np.linspace(low,high,3))
    cbar.ax.tick_params(labelsize=7)
    
    
    # Customize gridline settings
    gl = ax.gridlines(linewidth=1.5, linestyle='--', color='k')#,draw_labels=True)
    # Add equator gridline
    gl.ylocator = plt.FixedLocator([0])
    gl.xlines = False
    gl.ylines = True
    
    ax.coastlines()
    ax.add_feature(cfeature.LAND,facecolor='gray')
    
    
    # Add shelf break region
    lat_start, lat_end = -72, -70
    lon_start, lon_end = -115,-102
    num_vertices = 100
    # Generate vertices for the arc-shaped box
    box_lats = np.linspace(lat_start, lat_end, num_vertices)
    box_lons = np.linspace(lon_start, lon_end,num_vertices)
    west_edge_vertices = list(zip([lon_start]*num_vertices, box_lats))
    south_edge_vertices = list(zip(box_lons, [lat_start]*num_vertices))
    east_edge_vertices = list(zip([lon_end]*num_vertices, box_lats))
    north_edge_vertices = list(zip(box_lons, [lat_end]*num_vertices))
    vertices = west_edge_vertices + north_edge_vertices  + \
        east_edge_vertices[::-1] + south_edge_vertices[::-1]
    # Create a Polygon object from the vertices
    polygon = Polygon(vertices)
    # Plot the polygon
    ax.add_geometries([polygon], ccrs.PlateCarree(), edgecolor='red', \
                      linewidth=1.5,facecolor='none')
    
    
    # Plot proxy locations
    for proxy in proxy_dict:
        lon1 = proxy_dict[proxy][1]
        lat1 = proxy_dict[proxy][0]
        col = proxy_dict[proxy][2]
        if col == '#006d2c':
            tree=ax.plot(lon1,lat1, 'o',color=col, transform=ccrs.PlateCarree(),markersize=3.7,markeredgecolor='k',label='Tree rings',clip_on=True)
        elif col == 'magenta':
            cor=ax.plot(lon1,lat1, 'o',color=col, transform=ccrs.PlateCarree(), markersize=3.7,markeredgecolor='k',label='Corals',clip_on=True)
        else:
            ice=ax.plot(lon1,lat1, 'o',color=col, transform=ccrs.PlateCarree(), markersize=3.7,markeredgecolor='k',label='Ice cores',clip_on=True)
    


# Add legend to bottom of figure
# ax.legend([ice,cor,tree],['Ice cores','Corals','Tree rings'],\
#           ncol=3,bbox_to_anchor=(-.3, -.6),loc='lower center')
handles, labels = ax.get_legend_handles_labels()
unique_labels = list(set(labels))  # Get unique labels

# Create a legend with unique labels
leg = ax.legend([handles[labels.index(label)] for label in unique_labels], \
                unique_labels,loc='lower center',ncol=3,\
                    handlelength=.5,handletextpad=0.5,bbox_to_anchor=(.55,-.2),fontsize=7)



plt.subplots_adjust(left=0.05,right=0.85,top=0.98,bottom=0.1)
fig.text(0.06,0.93,'d)',fontsize=9,weight='bold')
fig.text(0.06,0.48,'e)',fontsize=9,weight='bold')

plt.savefig('Plots/Figure_1_d_e_proxy_locs.png',dpi=600)

# may want to crop white space before merging

#%% Merge with left panels of figure 1

from  PIL import Image

fig_a_c = Image.open('Plots/Figure_1_a_c_ASE_timeseries.png')
fig_d_e = Image.open('Plots/Figure_1_d_e_proxy_locs.png')

# create blank image with max width on page and height from both images
dpi = 600
merged_fig = Image.new('RGB', (fig_a_c.width+fig_d_e.width,max(fig_a_c.height,fig_d_e.height)),(255,255,255))

merged_fig.paste(fig_a_c, (0,0))
merged_fig.paste(fig_d_e, (fig_a_c.width,int(0.1*dpi)))

merged_fig.save('Plots/Figure_1_a_e.png',dpi=(600,600))
