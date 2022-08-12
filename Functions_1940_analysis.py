#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 17:55:37 2022

Contains all major functions used in analysis for 1940s paper. 
Includes retreiving data and stats functions. 

@author: gemma
"""

import xarray as xr
import numpy as np
from scipy import stats, signal
from mpl_toolkits import basemap

#%%
#LOADING TIME SERIES DATA------------------------------------------------------

#Load LMR reconstruction timeseries for one variable averaged over one region
def load_1d_recon(run,var,time_per,region,anom_ref=None):
    
    
    """
    Load LMR recon timeseries for specified variable in a specified region
    over specified period. Returns as numpy array.
    Returns SLP in hPa.

    Parameters
    ----------
    run : str
        Title of LMR run that matches the netCDF files
    var : str
        Variable name corresponding to var name in .nc file
        Ex: 'psl','u10','tas'
    time_per: list of 2 integers
        Start year, stop year of data to collect (inclusive)
    region : str
        Region over which to average
        Accepted: 'ASE' (see dictionary for locations)
    anom_ref: list with 2 ints, optional
        List with start,stop year for anomaly reference period
        Only needed if want to change ref period from 1961,1990
        *Or* if you want to make the anom reference period 1961 to 1990 have
        a mean value of 0. 

    Returns
    -------
    time: 1d np array
        Contains times associated with returned data
    data_reg: 1d np array
        1d array of data specified
    

    """
    #get region info
    loc_dict = {'ASE':[-72,-70,245,258]}
    lat1,lat2,lon1,lon2 = loc_dict[region]
    
    #get timing info
    start,stop = time_per
    
    #get specified data
    path = 'LMR_output/'+run+'_'+var+'.nc'
    ds = xr.open_dataset(path)
    ds = ds.sel(time=slice(start,stop),lat = slice(lat1,lat2),lon=slice(lon1,lon2))
    data = ds.get(var)
    data = np.squeeze(data) #has shape n_years,nlats,nlons
    time = ds.time
    
    #avg over region
    data_reg = np.mean(data,axis=1)
    data_reg = np.mean(data_reg,axis=1)
    
    #put in anomaly space relative to specified period
    if anom_ref:
        
        anom_start,anom_stop = anom_ref
        #calculate mean over new anom ref period and shift data by that amount
        os = np.mean(data_reg.sel(time = slice(anom_start,anom_stop)))
        data_reg = data_reg - os #subtracting gives you a mean of 0 for the new ref period
    
    #put in hPa if SLP
    if var == 'psl':
        data_reg = data_reg/100
    
    #format data
    data_reg = np.array(data_reg)
    time = np.array(time)
    
    return time, data_reg


#Load full ensemble of LMR reconstruction timeseries for one variable averaged over one region
def load_1d_recon_full_ens(run,var,time_per,region):
    
    
    """
    Load full ensemble of LMR recon timeseries for specified variable in a specified region
    over specified period. Returns as numpy array.
    Returns SLP in hPa.

    Parameters
    ----------
    run : str
        Title of LMR run that matches the netCDF files. must have full ens files!
    var : str
        Variable name corresponding to var name in .nc file
        Ex: 'psl','u10','tas'
    time_per: list of 2 integers
        Start year, stop year of data to collect (inclusive)
    region : str
        Region over which to average
        Accepted: 'ASE' (see dictionary for locations)

    Returns
    -------
    time: 1d np array
        Contains times associated with returned data
    data_reg: 2d np array
        2d array of data specified with shape n_years, n_ens
    

    """
    #get region info
    loc_dict = {'ASE':[-72,-70,245,258]}
    lat1,lat2,lon1,lon2 = loc_dict[region]
    
    #get timing info
    start,stop = time_per
    
    #get specified data
    path = 'LMR_output/Full_Ensemble/'+run+'_'+var+'_full.nc'
    ds = xr.open_dataset(path)
    ds = ds.sel(time=slice(start,stop),lat = slice(lat1,lat2),lon=slice(lon1,lon2))
    data = ds.get(var)
    #avg over region
    data_reg = np.mean(data,axis=1)
    data_reg = np.mean(data_reg,axis=1)
    time = ds.time

    #put in hPa if SLP
    if var == 'psl':
        data_reg = data_reg/100
    
    #format data
    data_reg = np.array(data_reg)
    time = np.array(time)
    
    return time, data_reg

def load_3d_recon(run,var,time_per,anom_ref = None):
    """
    Load map of recon data for specified time period.
    Option to change anom ref period.

    Parameters
    ----------
    run : str
        name of run corresponding to .nc files
    var : str
        name of var corresponding to .nc file name and contents
    time_per : list of 2 ints
        start, stop of timeseries (incl.)
    anom_ref : list of 2 ints, optional
        time period over which to remove the mean. The default is None.

    Returns
    -------
    time: 1d np array
        Contains times associated with returned data
    data_reg: 3d np array
        3d array of data specified with shape n_years, n_lats,nlons
    """

    
    #get specified data
    path = 'LMR_output/'+run+'_'+var+'.nc'
    ds = xr.open_dataset(path)
    start,stop = time_per
    ds = ds.sel(time=slice(start,stop))
    data = ds.get(var)
    data = np.squeeze(data) #has shape n_years,nlats,nlons
    data_shape = data.shape
    time = np.array(ds.time)
    
    #put in anomaly space relative to specified period (at every loc)
    if anom_ref:
        data_anom = np.zeros(data_shape)
        anom_start,anom_stop = anom_ref
        nlat = data_shape[1]
        nlon = data_shape[2]
        
        for lat_i in range(nlat):
            print("shifting offset for lat..",lat_i)
            for lon_j in range(nlon):
                data_loc = data[:,lat_i,lon_j]
                os_loc = np.mean(data_loc.sel(time = slice(anom_start,anom_stop)))
                data_anom_loc = data_loc - os_loc #subtracting gives you a mean of 0 for the new ref period
                data_anom[:,lat_i,lon_j] = data_anom_loc
        data = data_anom
    else:
        data = np.array(data)
        
    #put in hPa if SLP
    if var == 'psl':
        data = data/100
    
    
    return time, data

# Load 1d ERA5 timeseries avged over a region
def load_1d_era5(var,time_per,region,anom_ref=None):
    
    """
    Load ERA5 timeseries with specified variable, time period, and averaged over
    a region. 
    
    Parameters
    ----------
    var : str
        Variable name corresponding to var name in .nc file
        Ex: 'psl','u10000'
    time_per: list of 2 integers
        Start year, stop year of data to collect (inclusive)
    region : str
        Region over which to average
        Accepted: 'ASE' (see dictionary for locations)
    anom_ref: list with 2 ints, optional
        List with start,stop year for anomaly reference period
        Only needed if want to put in anomaly space 

    Returns
    -------
    time: 1d np array
        Contains times associated with returned data
    data_reg: 1d np array
        1d array of data specified. If no anon_ref specified, returns real values.

    """
    #get region info
    loc_dict = {'ASE':[-72,-70,245,258]}
    lat1,lat2,lon1,lon2 = loc_dict[region]
    
    #get timing info
    start,stop = time_per
    
    #get data
    path = 'Model/ERA5/annual_'+var+'_1979_2019.nc' 
    ds = xr.open_dataset(path)
    ds = ds.sel(time=slice(start,stop),lat=slice(lat2,lat1),lon=slice(lon1,lon2))
    data = ds.get(var) #shape n_years,n_lats,n_lons
    
    #average over lats and lons
    data_reg = np.mean(data,axis=1)
    data_reg = np.mean(data_reg,axis=1) #has shape n_years
    
    #put 1d array in anomaly space if specified
    if anom_ref:
        anom_start,anom_stop = anom_ref
        data_reg = data_reg - np.mean(data_reg.sel(time=slice(anom_start,anom_stop)))
    
    #put SLP in hPa
    if var == 'psl':
        data_reg = data_reg/100
    
    time = ds.time
    time = np.array(time)
    data_reg = np.array(data_reg)
    
    return time, data_reg

# Load 2d ERA5 timeseries (map of ERA5 data)
def load_3d_era5(var,time_per,anom_ref=None,asc_lats=False,regrid=None):
    
    """
    Load ERA5 timeseries with specified variable and time period
    
    Parameters
    ----------
    var : str
        Variable name corresponding to var name in .nc file
        Ex: 'psl','u10000'
    time_per: list of 2 integers
        Start year, stop year of data to collect (inclusive)
    anom_ref: list with 2 ints, optional
        List with start,stop year for anomaly reference period
        Only needed if want to put in anomaly space 
    asc_lats: bool, optional
        ERA5 has lats 90, 89.75, ..., -90. 
        If True, flips dataset so lats are increasing (to match recons)
        I.e., set True for calculating correlation maps.
    regrid: str, optional
        if you want to regrid ERA5 to the resolution of a recon, enter run name
        I.e., for calculating correlation maps

    Returns
    -------
    time: 1d np array
        Contains times associated with returned data
    data_reg: 3d np array
        3d array of data specified. If no anon_ref specified, returns real values.

    """
    
    #get timing info
    start,stop = time_per
    
    #get data
    path = 'Model/ERA5/annual_'+var+'_1979_2019.nc' 
    ds = xr.open_dataset(path)
    ds = ds.sel(time=slice(start,stop))
    if asc_lats:
        ds = ds.reindex(lat=ds.lat[::-1])
    data = ds.get(var) #shape n_years,n_lats,n_lons
    time = np.array(ds.time)
    
    
    if regrid: 
        print('Regridding...')
        data = np.array(data)
        era_lon = np.array(ds.lon)
        era_lat = np.array(ds.lat)
        
        #get recon grid info for interpolating era5
        ds_recon = xr.open_dataset('LMR_output/'+regrid+'_u10.nc')
        recon_lon,recon_lat = ds_recon.lon,ds_recon.lat
        new_lons, new_lats = np.meshgrid(recon_lon,recon_lat)
        
        #regrid ERA5 for every year
        data_regrid = []
        for year in range(len(data)):
            
            data_year = data[year,:,:]
            data_regrid_year = basemap.interp(data_year, era_lon, era_lat, \
                                                new_lons,new_lats, order=1)
            data_regrid.append(data_regrid_year)
        data = np.array(data_regrid)
        
        
    #put in anomaly space relative to specified period (at every loc)
    if anom_ref:
        print('Moving anomaly reference period...')
        data_anom = np.zeros(data.shape)
        anom_start,anom_stop = anom_ref
        anom_start_i = int(anom_start-time[0])
        anom_stop_i = int(anom_stop-time[0])+1
        nlat = data.shape[1]
        nlon = data.shape[2]
        
        for lat_i in range(nlat):
            print("shifting offset for lat..",lat_i)
            for lon_j in range(nlon):
                data_loc = data[:,lat_i,lon_j]
                os_loc = np.mean(data_loc[anom_start_i:anom_stop_i])
                data_anom_loc = data_loc - os_loc #subtracting gives you a mean of 0 for the new ref period
                data_anom[:,lat_i,lon_j] = data_anom_loc
        data = data_anom
    else:
        data = np.array(data)
        
    #put in hPa if SLP
    if var == 'psl':
        data = data/100
        
    return time, data
    

def load_1d_dal_recon(var,time_per,region,anom_ref=None):
    
    """
    Load Dalaiden recon timeseries for specified variable in a specified region
    over specified period. Returns as numpy array.
    Returns SLP in hPa.
    Anom ref period 1941-1990 unless specified.

    Parameters
    ----------
    var : str
        Variable name corresponding to var name in .nc file
        Ex: '850hpa-Uwind','SLP','SAT'
    time_per: list of 2 integers
        Start year, stop year of data to collect (inclusive)
    region : str
        Region over which to average
        Accepted: 'ASE' (see dictionary for locations)
    anom_ref: list with 2 ints, optional
        List with start,stop year for anomaly reference period
        Only needed if want to change ref period from 1941,1990

    Returns
    -------
    time: 1d np array
        Contains times associated with returned data
    data_reg: 1d np array
        1d array of data specified

    """
    
    #get region info
    loc_dict = {'ASE':[-72,-70,245,258]}
    lat1,lat2,lon1,lon2 = loc_dict[region]
    
    #get timing info
    start,stop = time_per
    
    path = 'Verification/Dalaiden_2021/'+var+'_ano_annual_recon-antarctic_1800-2000.nc'
    ds = xr.open_dataset(path,decode_times=False)
    ds = ds.sel(lat=slice(lat1,lat2),lon=slice(lon1,lon2))
    data = ds.get(var)
    
    #get time slice via indexing bc error with calendar
    start_idx = start - 1800
    stop_idx = stop - 1800 + 1
    data = data[start_idx:stop_idx] #shape n_years,n_lats,n_lons
    
    #average over region
    data_reg = np.mean(data,axis = 1)
    data_reg = np.mean(data_reg,axis = 1)
    data_reg = np.array(data_reg)
    
    #calculate anomalies relative to specified period
    if anom_ref:
        anom_start,anom_stop = anom_ref
        #calculate mean of data over new anom ref period and shift data by that much
        os = np.mean(data_reg[anom_start-start:anom_stop-start+1])
        data_reg = data_reg - os
    
    if var == 'SLP':
        data_reg = data_reg/100
    
    time = np.linspace(start,stop,stop-start+1)
    
    return time, data_reg

def load_1d_clim_model_concat_ens(model,vname,region,isolate_nat=False,adj_lens_em=False):
    """
    Get climate model data from full ensemble in LENS pi, LENS hist (add mpi later?)
    for a given variable and region. Does NOT put in anomaly space.
    
    If isolate_nat: remove LENS EM from each member (in LENS historical ONLY)
    
    If adj_lens_em: move the mean of LENS EM to match ERA5 (for isolating nat)
    
    Parameters
    ----------
    model : str
        which model to search 
        options: 'LENS_pi', 'LENS_hist', 'mpi_lm'
    vname : str
        'u10' or 'psl'
    region : str
        'ASE' or 'ASL'
        region to average over
    isolate_nat: bool
    adj_lens_em: bool

    Returns
    -------
    model_time: 1d nparray with model times. 
        for LENS historical, its 1920-2005 continued 40 times (until the year 5359)
    model_tseries_concat: 1d numpy array with model tseries
        for LENS historical, the 40 members are concatenated to make 1 long array.

    """
    if region == 'ASE':
        lat1,lat2 = -72,-70
        lon1,lon2 = 245,258
    elif region == 'ASL':
        lat1,lat2 = -75,-60
        lon1,lon2 = 180,310
       
    #get ensemble mean for putting in anomaly space----------------------------
    if model == 'LENS_hist':
        ens_mems = ['001','002','003','004','005','006','007','008','009','010',\
                '011','012','013','014','015','016','017','018','019','020',\
                '021','022','023','024','025','026','027','028','029','030',\
                '031','032','033','034','035','101','102','103','104','105']
        fname1 = 'Model/LENS/annual_'+vname+'_LENS_ens_'
        fname2 = '_1920_2005.nc'
        # n_model_yrs = len(ens_mems)*86
        print('concatenating 3440 years (40 ensembles from 1920 to 2005) in LENS historical')
        model_time = np.linspace(1920,1920+86*len(ens_mems)-1,86*len(ens_mems))
        
        #get LENS EM to remove from each member if you want to isolate the natural component
        if isolate_nat:
            print('removing LENS EM...')
            fname = 'Model/LENS/annual_'+vname+'_LENS_ens_mean_1920_2005.nc'
            ds = xr.open_dataset(fname)
            ds = ds.sel(lat = slice(lat1,lat2),lon=slice(lon1,lon2)) 
            em_data = ds.get(vname) #shape (86,2,11)
            if vname == 'psl':
                em_data = em_data/100
            #average over region
            em_reg = np.mean(em_data,axis=1)
            em_reg = np.mean(em_reg,axis=1)
            
            if adj_lens_em:
                print('Adjusting LENS EM so that it matches ERA5...')
                try:
                    era_time,era = load_1d_era5(vname,[1979,2005],region)
                except:
                    era_time,era = load_1d_era5('u1000',[1979,2005],region)
                
                #calculate offset in mean between ERA and LENS EM from 1979-2005
                os = np.mean(era) - np.mean(em_reg.sel(time=slice(1979,2005)))
                #shift LENS EM by the offset
                em_reg = em_reg + os
        
        
    elif model == 'LENS_pi':
        f_years = np.linspace(400,2100,(2100-400)//100+1)
        ens_mems_1 = [str(int(year)) for year in f_years]
        f_years_2 = np.linspace(499,2199,(2199-499)//100+1)
        ens_mems_2 = [str(int(year)) for year in f_years_2]
        ens_mems = [e1+'_'+e2 for e1,e2 in zip(ens_mems_1,ens_mems_2)]
        fname1 = 'Model/LENS_preindustrial/annual_'+vname+'_LENS_preindustrial_'
        fname2 = '.nc'
        # n_model_yrs = len(ens_mems) * 100 + 1 #+1 bc last member is 1 yr longer
        model_time = np.linspace(400,2200,2200-400+1)
        
        

    #get model tseries over region and put in anom space using value from above
    ens_tseries_list = []
    for ens in ens_mems:
        try:
            ds = xr.open_dataset(fname1 + ens + fname2)
        except:
            ds = xr.open_dataset(fname1 + '2100_2200.nc')
        ds = ds.sel(lat=slice(lat1,lat2),lon=slice(lon1,lon2))
        ens_data = ds.get(vname)
        if vname == 'psl':
            ens_data = ens_data/100
        #average over lats and lons
        ens_reg_data = np.mean(ens_data,axis=1)
        ens_reg_data = np.mean(ens_reg_data,axis=1)
        
        if isolate_nat:
            ens_reg_data_nat = ens_reg_data - em_reg
            ens_reg_data = ens_reg_data_nat
        
        ens_tseries_list.append(ens_reg_data)
        
    #make model data one big nparray
    model_tseries_concat = np.concatenate(ens_tseries_list)

        
    return model_time,model_tseries_concat

def load_1d_clim_model_em(model,vname,region):
    """
    Get ensemble mean of climate model data (set up for only LENS hist now)
    for a given variable and region. Does NOT put in anomaly space.
    
    Parameters
    ----------
    model : str
        which model to search 
        options: 'LENS_pi', 'LENS_hist', 'mpi_lm'
    vname : str
        'u10' or 'psl'
    region : str
        'ASE' or 'ASL'
        region to average over

    Returns
    -------
    model_time: 1d nparray with model times. 
        for LENS historical, its 1920-2005.
    model_tseries: 1d numpy array with model tseries
        for LENS historical, shape (86,)

    """
    if region == 'ASE':
        lat1,lat2 = -72,-70
        lon1,lon2 = 245,258
    elif region == 'ASL':
        lat1,lat2 = -75,-60
        lon1,lon2 = 180,310
       
    #get ensemble mean for putting in anomaly space----------------------------
    if model == 'LENS_hist':
        fname = 'Model/LENS/annual_'+vname+'_LENS_ens_mean_1920_2005.nc'
        ds = xr.open_dataset(fname)
        ds = ds.sel(lat = slice(lat1,lat2),lon=slice(lon1,lon2)) 
        em_data = ds.get(vname) #shape (86,2,11)
        if vname == 'psl':
            em_data = em_data/100
        #average over region
        em_reg = np.mean(em_data,axis=1)
        em_reg = np.mean(em_reg,axis=1)
        model_time = np.array(ds.time)
        
        return model_time, em_reg

def load_1d_clim_model_anom(model,vname,anom_ref,region):
    """
    
    Parameters
    ----------
    model : str
        which model to search 
        options: 'LENS_pi', 'LENS_hist', 'mpi_lm'
    vname : str
        'u10' or 'psl'
    anom_ref : list of 2 ints
        period (inclusive) over which to remove mean to put in anomaly space
    region : str
        'ASE' or 'ASL'
        region to average over

    Returns
    -------
    model_time: 1d nparray with model times. 
        for LENS historical, its 1920-2005 continued 40 times (until the year 5359)
    model_tseries_concat: 1d numpy array with model tseries
        for LENS historical, the 40 members are concatenated to make 1 long array.

    """
    if region == 'ASE':
        lat1,lat2 = -72,-70
        lon1,lon2 = 245,258
    elif region == 'ASL':
        lat1,lat2 = -75,-60
        lon1,lon2 = 180,310
       
    #get ensemble mean for putting in anomaly space----------------------------
    if model == 'LENS_hist':
        ens_mems = ['001','002','003','004','005','006','007','008','009','010',\
                '011','012','013','014','015','016','017','018','019','020',\
                '021','022','023','024','025','026','027','028','029','030',\
                '031','032','033','034','035','101','102','103','104','105']
        fname1 = 'Model/LENS/annual_'+vname+'_LENS_ens_'
        fname2 = '_1920_2005.nc'
        # n_model_yrs = len(ens_mems)*86
        print('concatenating 3440 years (40 ensembles from 1920 to 2005) in LENS')
        
        #get ensemble mean for anomaly calculation
        fname = 'Model/LENS/annual_'+vname+'_LENS_ens_mean_1920_2005.nc'
        ds = xr.open_dataset(fname)
        ds = ds.sel(time = slice(anom_ref[0],anom_ref[1]),lat = slice(lat1,lat2),\
                    lon=slice(lon1,lon2)) 
        em_data = ds.get(vname)
        if vname == 'psl':
            #LENS historical is huge
            em_data = em_data/100
        #average over region
        em_ref_per = np.mean(em_data,axis=1)
        em_ref_per = np.mean(em_ref_per,axis=1)
        #em_ref_per has shape (30,) for 30 times in reference period
        #avg over time
        em_ref_val = np.mean(em_ref_per)
        model_time = np.linspace(1920,1920+86*len(ens_mems)-1,86*len(ens_mems))
        
        print('removing ensemble mean value from',anom_ref,'calcuated from ensemble mean in LENS historical')
    
    elif model == 'LENS_pi':
        f_years = np.linspace(400,2100,(2100-400)//100+1)
        ens_mems_1 = [str(int(year)) for year in f_years]
        f_years_2 = np.linspace(499,2199,(2199-499)//100+1)
        ens_mems_2 = [str(int(year)) for year in f_years_2]
        ens_mems = [e1+'_'+e2 for e1,e2 in zip(ens_mems_1,ens_mems_2)]
        fname1 = 'Model/LENS_preindustrial/annual_'+vname+'_LENS_preindustrial_'
        fname2 = '.nc'
        # n_model_yrs = len(ens_mems) * 100 + 1 #+1 bc last member is 1 yr longer
        model_time = np.linspace(400,2200,2200-400+1)
        
        #select a 30-yr period which will be your anomaly reference for all years
        # ds = xr.open_dataset(fname1 + ens_mems[0] + fname2)
        # ds = ds.sel(time=slice(461,490),lat = slice(lat1,lat2),\
        #             lon=slice(lon1,lon2)) 
        ds = xr.open_dataset(fname1 + ens_mems[15] + fname2)
        ds = ds.sel(time=slice(anom_ref[0],anom_ref[1]),lat = slice(lat1,lat2),\
                    lon=slice(lon1,lon2)) 
        data = ds.get(vname)
        if vname == 'psl':
            data = data/100
        #average over region
        data_reg = np.mean(data,axis=1)
        data_reg = np.mean(data_reg,axis=1)
        em_ref_val = np.mean(data_reg,axis=0)
        
        print('removing ensemble mean value from',anom_ref,'calculated from the 1 ensemble member in LENS PI containing this period')
    
    #get model tseries over region and put in anom space using value from above
    ens_tseries_list = []
    for ens in ens_mems:
        try:
            ds = xr.open_dataset(fname1 + ens + fname2)
        except:
            ds = xr.open_dataset(fname1 + '2100_2200.nc')
        ds = ds.sel(lat=slice(lat1,lat2),lon=slice(lon1,lon2))
        ens_data = ds.get(vname)
        if vname == 'psl':
            ens_data = ens_data/100
        #average over lats and lons
        ens_reg_data = np.mean(ens_data,axis=1)
        ens_reg_data = np.mean(ens_reg_data,axis=1)
        #put in anom space relative to EM value for LENS hist, or relative to 1 mem for LENS PI
        ens_reg_anom = ens_reg_data - em_ref_val
        
        ens_tseries_list.append(ens_reg_anom)
        
    #make model data one big nparray
    model_tseries_concat = np.concatenate(ens_tseries_list)

        
    return model_time,model_tseries_concat

def calc_nino_idx(region,version,time_per):
    """
    Calculates Nino Index in specified region (i.e. 3.4) using ERSSTv3b or v5.
    Averages over specified nino region and removes mean.

    Parameters
    ----------
    region : str
        Nino region. Accepted: Nino3.4, Nino3, Nino4, Nino1_2
    version : str
        Version of ERSST to use. Accepted: v3b or v5
    time_per : list of 2 ints
        First and last year over data to get (inclusive)

    Returns
    -------
    time : 1d numpy array
        times associated with returned index
    data_reg : 1d numpy array
        Timeseries of nino index

    """
    #get details
    region_dict = {'Nino3.4':[-5,5,190,240],'Nino3':[-5,5,210,270],\
               'Nino4':[-5,5,160,210],'Nino1_2':[-10,0,270,280]}
    lat1,lat2,lon1,lon2 = region_dict[region]
    start,stop = time_per
    
    #load data
    path = 'Verification/annual_ersst'+version+'_1854_2019.nc'
    ds = xr.open_dataset(path)
    ds = ds.sel(time=slice(start,stop),lat = slice(lat2,lat1),\
                        lon = slice(lon1,lon2))
    data = ds.sst 
    time = ds.time
    
    #avg over region and calulate index
    data_lat_mean = data.mean(dim = 'lat')
    data_reg = data_lat_mean.mean(dim = 'lon')
    data_reg = data_reg - np.mean(data_reg)
    data_reg = data_reg[0:-1] #last value is funny
    time = time[0:-1]
    
    #format data
    time = np.array(time)
    data_reg = np.array(data_reg)
    
    return time, data_reg

#STATS CALCULATIONS-----------------------------------------------------------

# Calculate correlations between 2 timeseries
def calc_1d_corr(x,y,return_format = 'string'):
    """
    Calculates correlations between 2 numpy arrays x and y. 
    Accounts for atuocorrelation in assessment of significance. 
    Tests for significance with 95% confidence.
    Returns strings. Designed for print statements and text labels in plots. 

    Parameters
    ----------
    x : 1d numpy array
    y : 1d numpy array of same length as x
    return_format : str, optional
        default returns string of corr to 2 decimal places, and '*' for sig p-val
        alternative is a 'float' of corr and p-val

    Returns
    -------
    corr_str : str
        Pearson r correlation with 2 decimal points
    sig : str
        Returns '*' if significance with 95% confidence (p-val<0.05).
        Else ''

    """
    
    #calculate sample correlation
    corr = stats.pearsonr(x,y)[0]
    if return_format == 'string':
        corr_formatted = "{:.2f}".format(corr)
    else:
        corr_formatted = corr
    
    #calculate number of independent obs n_eff
    autocorr_x = abs(stats.pearsonr(x[0:-1],x[1:])[0])
    autocorr_y = abs(stats.pearsonr(y[0:-1],y[1:])[0])
    n = len(x)
    n_eff = n * (1-autocorr_x * autocorr_y)/(1+autocorr_x * autocorr_y)
    
    #do t-test
    dof = n_eff -2
    t_stat = corr * np.sqrt(dof/(1-corr**2))
    p = stats.t.sf(np.abs(t_stat), dof)*2 
    #p_string = "{:.3f}".format(p)
    if p < 0.05:
        sig = '*'
    else:
        sig = ''
    if return_format == 'float':
        sig = p
    
    return corr_formatted, sig

# Calculate coefficient of efficiency between 2 timeseries
def calc_1d_ce(x,v,return_format = 'string'):
    """
    Calculates coefficient of efficiency between a test 1d array and a 
    verification 1d array.

    Parameters
    ----------
    x : 1d numpy array
        Test dataset (i.e., reconstruction)
    v : 1d numpy array of same length as x
        Verification dataset (i.e., ERA5)
    return_format: string, optional
        default is string, other option is 'float'

    Returns
    -------
    ce_str : str or float, depending on return_format
        CE value (if string, with 2 decimal places)

    """
    #x = recon, v = verif
    
    error = v - x
    
    # CE following equation 7 in Hakem et al., 2016
    numer = np.sum(np.power(error,2))
    denom = np.sum(np.power((v-np.mean(v)),2))
    CE    = 1 - (numer/denom)
    
    if return_format == 'string':
        ce_formatted = '{:.2f}'.format(CE)
    elif return_format == 'float':
        ce_formatted = CE
    
    return ce_formatted

#calculate 95% conf interval for indpendent data
def calc_conf_int_ind_data(data,alpha):
    """
    Calculate confidence interval for  independent data set
    for not independent dataset, need to calculate n_eff accounting for autocorrelation.
    
    Parameters
    ----------
    data : 1d np array
    alpha: float
        0.05 = 95% conf
    Returns
    -------
    m: float
        mean of data
    [lower,upper[: list of 2 floats]
        lower, upper bound of ci


    """
    m = data.mean() 
    s = data.std() 
    dof = len(data)-1 
    t_crit = np.abs(stats.t.ppf((alpha)/2,dof))
    
    lower = m - s*t_crit / np.sqrt(len(data))
    upper = m + s*t_crit / np.sqrt(len(data))
    
    
    return m, [lower, upper]
