#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:34:08 2021

Makes 2 plots:
1. slp OR u10 anomalies in a given year in pace for each ensemble member
2. slp or u10 anomalies in ensemble mean of pacemaker models for multiple years. 
   compares all 3 models.

@author: gemma
"""
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

#for pacemaker comparison you can choose map area
map_area = 'South Pacific' #SH or South Pacific or 'global'
lon_labs = False #whether to plot labels for longitudes on bottom row of pacemaker comparison
save_fig = False

#%% Plot multiple years in a single pacemaker sim, all ens mems (bonus map, not in paper)

fs = 12
year = 1938
vname = 'psl' #psl only for indian and atlantic
model ='IOD_PACE' #PACE, IOD_PACE, ATL_PACE


if model == 'PACE':
    ens_list = ['01','02','03','04','05','06','07','08','09','10',\
            '11','12','13','14','15','16','17','18','19','20']
    fname1 = '../Model/PACE/annual_PACE_' + vname + '_ens_'
elif model == 'IOD_PACE':
    ens_list = ['01','02','03','04','05','06','07','08','09','10']
    fname1 = '../Model/IOD_PACE/annual_'+vname+'_IOD_PACE_ens_'
elif model == 'ATL_PACE':
    ens_list = ['01','02','03','04','05','06','07','08','09','10']
    fname1 = '../Model/ATL_PACE/annual_'+vname+'_ATL_PACE_ens_'

ens_data = []
for ens in ens_list:
    
    fname2 = '_1920_2005.nc'
    ds = xr.open_dataset(fname1 + ens + fname2)
    data = ds.get(vname)
    
    #put in anom space from 61-90
    #this calculates the mean at every location and subtracts it from every location
    #FIX: need to replace with the ensemble mean
    data_anom = data - np.mean(data[41:71,:,:],axis=0)
    data_year = data_anom.sel(time = year)
    if vname == 'psl':
        data_year = data_year/100
    
    ens_data.append(data_year)
    
lon,lat = ds.lon,ds.lat

#plot anomalies in full ensemble of model in a given year

def make_map():
    #plot maps-----------------------
    n_cols = len(ens_list)//5
    fig = plt.figure()
    plot_width = 2.8 * (n_cols//2) 
    fig.set_size_inches(5.6,6)
    
    
    for i in range(len(ens_list)):
        
        print(i)
        #plot_num goes from left to right, top to bottom. Does SLP figs first. 
        fig.add_subplot(5,n_cols,i+1)
                                   
        # map = Basemap(projection='gall',llcrnrlat=-76,urcrnrlat=-60,\
        #                   llcrnrlon=220,urcrnrlon=290,resolution='c')
        map = Basemap(projection='gall',llcrnrlat=-80,urcrnrlat=-30,\
                          llcrnrlon=150,urcrnrlon=320,resolution='c')
            
        if i %2 == 0:
            map.drawparallels(np.arange(-90,90,20),labels=[True,False,False,False],fontsize=fs-4,linewidth=0.5)
    
        else:
            map.drawparallels(np.arange(-90,90,20),labels=[False,False,False,False],fontsize=fs-4,linewidth=0.5)
        map.drawmeridians(np.arange(0,360,50),labels=[False,False,False,True],fontsize=fs-4,linewidth=0.5)
        if vname == 'u10':
            levs = np.linspace(-1,1,9)
        else:
            plim = 3
            levs = np.linspace(-plim,plim,9)
        x,y = map(*np.meshgrid(lon,lat))
        #pcolormesh
        data =  ens_data[i]
        mp=map.contourf(x[1:],y[1:],data[1:],cmap='RdBu_r',extend='both',levels=levs)#,vmin=low,vmax=high) 
        map.drawcoastlines()
        map.drawmapboundary()
        map.fillcontinents()
        plt.title('Ensemble ' + ens_list[i], fontsize=fs-2, pad=-.5)
            
        if vname == 'u10':
            #add box over ASE
            lat1,lat2,lon1,lon2 = -72,-70,245,258
            x1,y1 = map(lon2,lat1) #lower right lon (neg lon means west),lower right lat
            x2,y2 = map(lon2,lat2) #upper right lon, lat
            x3,y3 = map(lon1,lat2) #upper left
            x4,y4 = map(lon1,lat1)
            poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],fill=False, \
                                edgecolor='black',linewidth=1)
            plt.gca().add_patch(poly)
            
    
    cb_ax_u = fig.add_axes([0.35, 0.07, 0.3, 0.02]) #lower left corner position, axis width, axis height
    cb_u = fig.colorbar(mp, cax=cb_ax_u, extend='both',orientation = 'horizontal')
    
    if vname == 'u10':
        cb_u.set_ticks([-1,0,1])
        cb_u.set_ticklabels([-1,0,1])
        cb_u.set_label(label = r'U$_S$ (m/s)', fontsize=fs-1, labelpad = -.5)
    else:
        cb_u.set_ticks([-plim,0,plim])
        cb_u.set_ticklabels([-plim,0,plim])
        cb_u.set_label(label = 'SLP (hPa)', fontsize=fs-1, labelpad = -.5)
    cb_u.ax.tick_params(labelsize=fs-2)
    plt.subplots_adjust(wspace=0.1,hspace=0.4,top = 0.92,bottom=0.12,right=0.985,left=0.055)
    fig.text(0.39,0.96,model+' '+str(year),fontsize=14)
    
    return

make_map()
    
#plt.savefig('Plots/Figure_x_'+str(year)+'_PACE_model_maps_'+vname+'.png', bbox_inches = 'tight',dpi=600) #setting the dpi changes the size you specified above

#%% Figure 6 calcs - plot anoamlies in ensemble mean for all pacemakers for one variable

# Get pacemaker EM data in anomalies and 2sigma values of the EM

def calc_pace_anoms_signif(model_name, vname):
    """
    Get pacemaker ensemble mean data, put in anomaly space relative to 1961-1990
    Calculate variance of EM data and 2 sigma value at each gridpoint
    Create mask for anomalies less than 2 sigma

    Parameters
    ----------
    model_name : str
        which pacemaker simulation (keys in fname_dict)
    vname : str
        u10, psl, tas

    Returns
    -------
    em_anom_data : 3d xr data array from 1920-2005 (n_times, n_lats, n_lons)
    em_std : 2d xr data array  (n_lats, n_lons)
        1 std of the em (for masking single years of data in plots)

    """

    fname_dict = {'Pacific':'../Model/PAC_PACE/annual_'+vname+'_PAC_PACE_ens_mean_1920_2005.nc',\
                  'Indian':'../Model/IOD_PACE/annual_'+vname+'_IOD_PACE_ens_mean_1920_2005.nc',\
                  'Atlantic':'../Model/ATL_PACE/annual_'+vname+'_ATL_PACE_ens_mean_1920_2005.nc'}
    
    ds = xr.open_dataset(fname_dict[model_name])
    em_data = ds.get(vname)
    
    # Put in anomaly space
    em_data_anom = em_data - np.mean(em_data[41:71,:,:],axis=0)
    
    if vname == 'psl':
        em_data_anom = em_data_anom/100
    
    #Calculate std dev
    em_std = np.std(em_data_anom, axis=0)

    return em_data_anom, em_std
    
    
pac_pace_anom, pac_pace_std = calc_pace_anoms_signif('Pacific', vname)
iod_pace_anom, iod_pace_std = calc_pace_anoms_signif('Indian', vname) 
atl_pace_anom, atl_pace_std = calc_pace_anoms_signif('Atlantic', vname)




#%% Make figure 6

start,stop = 1938,1942
years = np.linspace(start,stop,stop-start+1,dtype=int)
fs = 10

def make_model_comp_map():
    
    
    #plot maps-----------------------
    fig = plt.figure()
    n_years = len(years)
    fig.set_size_inches(2.65,4)
    
    # Define plotting details
    meta_dict = {'psl':[3,'SLP anomaly (hPa)','RdBu_r',8.5e6,5.5e6,-60,20],\
                 'u10':[.7,r'U$_S$ anomaly (m/s)','PuOr_r',3.5e6,2.3e6,-71,10],\
                 'tas':[1.5,'TAS anomaly (°C)','PiYG_r',2.7e6,1.2e6,-70.5,5]}
    lim, lab, cm, map_width, map_height, lat0, n_parll = meta_dict[vname]


    #plot 3 model results for each year
    for yr_idx in range(n_years):
        
        year = years[yr_idx]
        print(yr_idx,year)
        #plot_num goes from left to right, top to bottom. 
        
        #Pacific PACE plot for this year
        fig.add_subplot(n_years,3,yr_idx*3+1)
        if map_area == 'South Pacific':                           
            map = Basemap(projection='lcc',lat_0=lat0,lon_0=250,\
                      lat_1=-60,lat_2=-50,resolution='c',width=map_width,height=map_height)
            map.drawparallels(np.arange(-90,90,n_parll),labels=[True,False,False,False],fontsize=fs-4,linewidth=0.5)
        elif map_area == 'SH':
            map = Basemap(projection='gall',llcrnrlat=-80,urcrnrlat=15,\
                      llcrnrlon=0,urcrnrlon=360,resolution='c')
            map.drawparallels(np.arange(-90,90,20),labels=[True,False,False,False],fontsize=fs-4,linewidth=0.5)
        elif map_area == 'global':
            map = Basemap(projection='gall',llcrnrlat=-80,urcrnrlat=85,\
                      llcrnrlon=0,urcrnrlon=360,resolution='c')
            map.drawparallels(np.arange(-90,90,30),labels=[True,False,False,False],fontsize=fs-4,linewidth=0.5)
        
        if yr_idx == n_years - 1:
            map.drawmeridians(np.arange(0,360,50),labels=[False,False,False,lon_labs],fontsize=fs-4,linewidth=0.5)
        else:
            map.drawmeridians(np.arange(0,360,50),labels=[False,False,False,False],fontsize=fs-4,linewidth=0.5)
        
        
        levs = np.linspace(-lim,lim,9)
        x,y = map(*np.meshgrid(lon,lat))
        
        # Plot anom data 
        pac_pace_anom_yr = pac_pace_anom.sel(time=year)
        
        # # plot anom data 
        mp = map.contourf(x[1:],y[1:],pac_pace_anom_yr[1:], levels=levs, cmap=cm,extend='both')
        
        # Uncomment to plot only signific vals
        # Create a new array with anom values where mask is True, and fill with NaN where mask is False
        pac_sig_data = np.where(abs(pac_pace_anom_yr) > abs(pac_pace_std), pac_pace_anom_yr, np.nan)
        map.contourf(x[1:],y[1:],pac_sig_data[1:], levels=levs, hatches=['..'],cmap=cm,extend='both')
        
        # Uncomment to contour vals > 1 sig
        # pac_mask = abs(pac_pace_anom_yr) < abs(pac_pace_std)
        # map.contour(x[1:],y[1:],pac_mask[1:], colors='gray', linestyles='-',linewidths=0.2)


        map.drawcoastlines()
        map.drawmapboundary()
        map.fillcontinents()
        if vname == 'u10':
            #add box over ASE
            lat1,lat2,lon1,lon2 = -72,-70,245,258
            x1,y1 = map(lon2,lat1) #lower right lon (neg lon means west),lower right lat
            x2,y2 = map(lon2,lat2) #upper right lon, lat
            x3,y3 = map(lon1,lat2) #upper left
            x4,y4 = map(lon1,lat1)
            poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],fill=False, \
                                edgecolor='black',linewidth=1)
            plt.gca().add_patch(poly)
            
        
        #Indian PACE plot for this year
        fig.add_subplot(n_years,3,yr_idx*3+2)
        if map_area == 'South Pacific':           
            map = Basemap(projection='lcc',lat_0=lat0,lon_0=250,\
                      lat_1=-60,lat_2=-50,resolution='c',width=map_width,height=map_height)
            map.drawparallels(np.arange(-90,90,n_parll),labels=[False,False,False,False],fontsize=fs-4,linewidth=0.5)
        elif map_area == 'SH':
            map = Basemap(projection='gall',llcrnrlat=-80,urcrnrlat=15,\
                      llcrnrlon=0,urcrnrlon=360,resolution='c')
            map.drawparallels(np.arange(-90,90,20),labels=[True,False,False,False],fontsize=fs-4,linewidth=0.5)
        elif map_area == 'global':
            map = Basemap(projection='gall',llcrnrlat=-80,urcrnrlat=85,\
                      llcrnrlon=0,urcrnrlon=360,resolution='c')
            map.drawparallels(np.arange(-90,90,30),labels=[False,False,False,False],fontsize=fs-4,linewidth=0.5)
        
        if yr_idx == n_years - 1:
            map.drawmeridians(np.arange(0,360,50),labels=[False,False,False,lon_labs],fontsize=fs-4,linewidth=0.5)
        else:
            map.drawmeridians(np.arange(0,360,50),labels=[False,False,False,False],fontsize=fs-4,linewidth=0.5)
        x,y = map(*np.meshgrid(lon,lat))
        
        # Contour vals > 1 sig and 2 sig
        iod_pace_anom_yr = iod_pace_anom.sel(time=year)

        mp = map.contourf(x[1:],y[1:],iod_pace_anom_yr[1:],cmap=cm,extend='both',levels=levs)#,vmin=low,vmax=high)
        
        iod_sig_data = np.where(abs(iod_pace_anom_yr) > abs(iod_pace_std), iod_pace_anom_yr, np.nan)
        map.contourf(x[1:],y[1:],iod_sig_data[1:], levels=levs, hatches=['..'],cmap=cm,extend='both')
        

        
        map.drawcoastlines()
        map.drawmapboundary()
        map.fillcontinents()
        plt.title(str(year), fontsize=fs-1, pad=-.5)
        if vname == 'u10':
            #add box over ASE
            lat1,lat2,lon1,lon2 = -72,-70,245,258
            x1,y1 = map(lon2,lat1) #lower right lon (neg lon means west),lower right lat
            x2,y2 = map(lon2,lat2) #upper right lon, lat
            x3,y3 = map(lon1,lat2) #upper left
            x4,y4 = map(lon1,lat1)
            poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],fill=False, \
                                edgecolor='black',linewidth=1)
            plt.gca().add_patch(poly)
            
            
        
        #Atlantic PACE plot for this year
        fig.add_subplot(n_years,3,yr_idx*3+3)
        if map_area == 'South Pacific':
            map = Basemap(projection='lcc',lat_0=lat0,lon_0=250,\
                      lat_1=-60,lat_2=-50,resolution='c',width=map_width,height=map_height)
            map.drawparallels(np.arange(-90,90,n_parll),labels=[False,False,False,False],fontsize=fs-4,linewidth=0.5)
        elif map_area == 'SH':
            map = Basemap(projection='gall',llcrnrlat=-80,urcrnrlat=15,\
                      llcrnrlon=0,urcrnrlon=360,resolution='c')
            map.drawparallels(np.arange(-90,90,20),labels=[True,False,False,False],fontsize=fs-4,linewidth=0.5)
        elif map_area == 'global':
            map = Basemap(projection='gall',llcrnrlat=-80,urcrnrlat=85,\
                      llcrnrlon=0,urcrnrlon=360,resolution='c')
            map.drawparallels(np.arange(-90,90,30),labels=[False,False,False,False],fontsize=fs-4,linewidth=0.5)
            map.drawmeridians(np.arange(0,360,50),labels=[False,False,False,False],fontsize=fs-4,linewidth=0.5)
        
        if yr_idx == n_years - 1:
            map.drawmeridians(np.arange(0,360,50),labels=[False,False,False,lon_labs],fontsize=fs-4,linewidth=0.5)
        else:
            map.drawmeridians(np.arange(0,360,50),labels=[False,False,False,False],fontsize=fs-4,linewidth=0.5)
        x,y = map(*np.meshgrid(lon,lat))
        
        # Plot anom data 
        atl_pace_anom_yr = atl_pace_anom.sel(time=year)
        mp=map.contourf(x[1:],y[1:],atl_pace_anom_yr[1:],cmap=cm,extend='both',levels=levs)
        
        atl_sig_data = np.where(abs(atl_pace_anom_yr) > abs(atl_pace_std), atl_pace_anom_yr, np.nan)
        map.contourf(x[1:],y[1:],atl_sig_data[1:], levels=levs, hatches=['..'],cmap=cm,extend='both')
        
        
        map.drawcoastlines()
        map.drawmapboundary()
        map.fillcontinents()
        
        if vname == 'u10':
            #add box over ASE
            lat1,lat2,lon1,lon2 = -72,-70,245,258
            x1,y1 = map(lon2,lat1) #lower right lon (neg lon means west),lower right lat
            x2,y2 = map(lon2,lat2) #upper right lon, lat
            x3,y3 = map(lon1,lat2) #upper left
            x4,y4 = map(lon1,lat1)
            poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],fill=False, \
                                edgecolor='black',linewidth=1)
            plt.gca().add_patch(poly)
        
            
    
    if vname == 'tas':
        cb_ax_u = fig.add_axes([0.36, 0.07, 0.3, 0.015]) #height of .018 for slp, u10
    else:
        cb_ax_u = fig.add_axes([0.38, 0.09, 0.3, 0.018])
    cb_u = fig.colorbar(mp, cax=cb_ax_u, extend='both',orientation = 'horizontal')
    
    cb_u.set_ticks([-lim,0,lim])
    cb_u.set_ticklabels([-lim,0,lim])
    cb_u.set_label(label = lab, fontsize=fs-2,labelpad=-0.5)
    cb_u.ax.tick_params(labelsize=fs-3)

    if map_area == 'South Pacific':
        plt.subplots_adjust(wspace=0.05,hspace=0.4,top = 0.92,bottom=0.13,right=0.99,left=0.07)
        if vname == 'psl':
            fig.text(0.09,0.96,'a) Pacific',fontsize=10)
            fig.text(0.415,0.96,'b) Indian',fontsize=10)
            fig.text(0.71,0.96,'c) Atlantic',fontsize=10)
        elif vname == 'u10':
            fig.text(0.09,0.96,'d) Pacific',fontsize=10)
            fig.text(0.415,0.96,'e) Indian',fontsize=10)
            fig.text(0.71,0.96,'f) Atlantic',fontsize=10)
    elif map_area == 'SH':
        fig.set_size_inches(8.5,6)
        plt.subplots_adjust(wspace=0.15,hspace=0.6,top = 0.92,bottom=0.10,right=0.99,left=0.04)
        fig.text(0.145,0.96,'Pacific',fontsize=10)
        fig.text(0.48,0.96,'Indian',fontsize=10)
        fig.text(0.8,0.96,'Atlantic',fontsize=10)
    elif map_area == 'global':
        fig.set_size_inches(5,5.8)
        plt.subplots_adjust(wspace=0.07,top = 0.93,bottom=0.12,right=0.99,left=0.06)
        fig.text(0.145,0.96,'a) Pacific',fontsize=10)
        fig.text(0.465,0.96,'b) Indian',fontsize=10)
        fig.text(0.77,0.96,'c) Atlantic',fontsize=10)
        
    return

make_model_comp_map()

if save_fig:
    plt.savefig('Plots/Figure_pacemaker_comparisons_1940_event_'+vname+'_'+map_area+'.png',dpi=400)

#%% Merge with psl and us panels into 1 fig

from  PIL import Image

fig_a = Image.open('Plots/Figure_pacemaker_comparisons_1940_event_psl_South Pacific.png')
fig_b = Image.open('Plots/Figure_pacemaker_comparisons_1940_event_u10_South Pacific.png')

# create blank image with max width on page and height from both images
dpi=400
merged_fig = Image.new('RGB', (fig_a.width+fig_b.width,fig_a.height),(255,255,255))

merged_fig.paste(fig_a, (0,0))
merged_fig.paste(fig_b, (fig_a.width,0))

merged_fig.save('Plots/Figure_x_all_pacemakers_psl_u10.png',dpi=(400,400))
