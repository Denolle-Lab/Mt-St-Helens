import numpy as np
import pandas as pd
import obspy
import obspy.signal.filter
from obspy import UTCDateTime
import datetime
import scipy
import glob
import sys
import os
import math
import matplotlib.pyplot as plt
import matplotlib

sys.path.append("/data/wsd01/pnwstore/")
from pnwstore.mseed import WaveformClient
client = WaveformClient()

sys.path.append('/home/koepflma/project1/Mt-St-Helens')
from functions import *

def main_noise(jday, year):
    
    # define some parameters------------------------------------------------------------------------------------------------------
    # station parameters
    net = 'UW'
    sta = 'SHW'
    cha = 'EHZ'
    
    # instrument response parameters
    pre_filt = [1e-2, 5e-2, 45, 50]
    water_level = 60
    
    # time windows
    win_len = 2**16 # number of points in window
    win_overlap = 0 # number of overlapping points
    min_am = 0.8 # minimal amount of datapoints in sliced trace to take trace into acount
    
    # saving array
    save_path = 'first_test/{}/{}'.format(year,sta) # path where to save file
    save_filename = '{}_{}_{}'.format(year, jday, sta) # file name
    
    # read in stream--------------------------------------------------------------------------------------------------------------
    try:
        st_r = read_stream(net, sta, cha, year, jday)
    except:
#        print('Problem during reading mseed to stream: {}-{}'.format(year,day))
        return # continue with next day, everything below is skipped
        
    st = st_r#.copy() # no copy due to memory
    
    # correct insrument response--------------------------------------------------------------------------------------------------
    inv = obspy.read_inventory('/auto/pnwstore1-wd11/PNWStationXML/{}/{}.{}.xml'.format(net,net,sta))
    for tr in st:
        s_time_str = str(tr.stats['starttime']).split('.')[0].replace(':', '-')
        tr.remove_response(inventory=inv, zero_mean=True,taper=True, taper_fraction=0.05,
                              pre_filt=pre_filt, output="VEL", water_level=water_level,
                              plot=False)
#                               plot='sensor_response_tests/{}__pre_filt{}-{}_{}-{}__water_level{}.png'.format(
#                               s_time_str,
#                               pre_filt[0], pre_filt[1], pre_filt[2], pre_filt[3], water_level))
    
    
    # calculate PSD---------------------------------------------------------------------------------------------------------------
    Pxx_list =  [] # initialize list
    # calculate psd for long enought traces and weight the trace depending on the length of the trace
    for tr in st:
        if len(tr.data) >= win_len: # trace is as long or longer as window length
            try:
                Pxx, freqs = matplotlib.mlab.psd(tr.data, NFFT=win_len, noverlap=win_overlap, Fs=st[0].stats.sampling_rate) # PSD
                Pxx = Pxx[(freqs>1e-1) & (freqs<2e1)] # save only between 0.1-20Hz
                Pxx_list.append(Pxx)
#                print(Pxx.shape, freqs.shape, len(Pxx_list))
            except:
#                print('Long trace, but problems during psd calculations: {}'. format(tr))
                pass
        else: # trace is shorter than window length
#            print('Short trace: {}'. format(tr))
            pass

    # calculate mean PSD over one day between traces
    Pxx = np.mean(Pxx_list, axis=0)
    freqs = freqs[(freqs>1e-1) & (freqs<2e1)] # save only between 0.1-20Hz
    
    # create array with starttimes for one day (UTCDateTime)----------------------------------------------------------------------
    start_times = np.arange(UTCDateTime(st[0].stats['starttime'].date), # midnight of start day (UTCDateTime)
                            UTCDateTime(st[0].stats['starttime'].date)+60*60*24, # midnight of next day (UTCDateTime)
                            (win_len-win_overlap)/st[0].stats['sampling_rate'])[:-1] # time steps in seconds
            
            
    # merge traces within a stream------------------------------------------------------------------------------------------------
    st_merge = st.copy()
    tr_merge = st_merge.merge()[0]

    # initialize lists------------------------------------------------------------------------------------------------------------
    rms_list = []
    rmes_list = []
    pgv_list = []
    pga_list = []

    # Calculate RMS, RMeS, PGV and PGA--------------------------------------------------------------------------------------------
    # loop over starttimes within one day
    for s_time in start_times:

        # try to cut the trace and calculate RMS, RMeS, PGV and PGA
        try:
            tr_cut = tr_merge.slice(s_time, s_time + win_len/tr.stats['sampling_rate']) # win_len in sec
            tr_cut = tr_cut.data

            # if trace is long enought calculate RMS, RMeS, PGV and PGA
            if len(tr_cut) >= min_am*win_len:
                rms = np.sqrt(np.mean(tr_cut**2))
                rmes = np.sqrt(np.median(tr_cut**2))
                pgv = max(abs(tr_cut))

                tr_acc = (tr_cut.copy()[:-1] - tr_cut.copy()[1:]) /tr.stats['delta']
                pga = max(abs(tr_acc))

            else:
#                print('Trace too short: {}'.format(tr_cut))
                rms = np.nan
                rmes = np.nan
                pgv = np.nan
                pga = np.nan

            # append RMS, RMeS, PGV and PGA to list
            rms_list.append(rms)
            rmes_list.append(rmes)
            pgv_list.append(pgv)
            pga_list.append(pga)   

        except:
#            print('Problem at starttime: {}'.format(s_time))
            pass
            
    # convert lists into arrays---------------------------------------------------------------------------------------------------
#     rms_ar = np.array(rms_list)
#     rmes_ar = np.array(rmes_list)
#     pgv_ar = np.array(pgv_list)
#     pga_ar = np.array(pga_list)

#     day_ar = np.array([freqs, Pxx, start_times, rms_ar, rmes_ar, pgv_ar, pga_ar])
    day_ar = np.array([freqs, Pxx, start_times, rms_list, rmes_list, pgv_list, pga_list])


    # initialize save path and save array-----------------------------------------------------------------------------------------
    if not os.path.exists(save_path): # create folders from save_path if not exists
        os.makedirs(save_path)
        
    save_nparray(save_path, save_filename, day_ar) # save array
    
    return()


import multiprocessing
from functools import partial

# multiprocessing---------------------------------------------------------------------
p = multiprocessing.Pool(processes=4)
p.map(partial(main_noise,  year=2018), range(1,16))
p.close()
p.join()