import numpy as np
import pandas as pd
import obspy
import obspy.signal.filter
import datetime
import scipy
from scipy.fft import fft
import glob
import sys
import os
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.append("/data/wsd01/pnwstore/")
from pnwstore.mseed import WaveformClient
client = WaveformClient()

# read one mseed file (1 day) from pnwstore as stream ------------------

def read_stream(net, sta, cha, year, jday):
    
    ''' 
    net: network name as string
    sta: station name as string
    cha: channel name as string
    year: year as int
    jday: julian-day as int
    
    returns: sream (one day long)
    '''
    try:
        st = obspy.Stream()
        st_read = client.get_waveforms(network='{}'.format(net), station='{}'.format(sta), channel='{}'.format(cha), 
                                       year='{}'.format(year), doy='{}'.format(jday))
        st += st_read
        
    except:
        print('pass: {}.{}.{} {}-{}'.format(net, sta, cha, year, jday))
        
    return(st)
# preprocessing --------------------------------------------------------------------
def detrend_taper_stream(st, det_type, max_per): # useless
    
    '''
    st: stream
    det_type: see link, string
        https://docs.obspy.org/master/packages/autogen/obspy.core.trace.Trace.detrend.html#obspy.core.trace.Trace.detrend
    max_per: decimal percentage of taper at one end (ranging from 0. to 0.5) as float
    '''
    
    return
    
def rem_instr_resp():
    return()
    

def read_taper_stream(sta,year,jday):
    st = obspy.Stream()
    st_d = obspy.Stream()
    try:
        # this stream will be used for RSAM and DSAR calculations
        #st_read = obspy.read('/media/manuela/T7/data/wd05/PNW{}/UW/{}/{}/EDM.UW.{}.{}'.format(year,year,jday,year,jday))
        st_read = client.get_waveforms(network='UW', station='{}'.format(sta), channel='*', year='{}'.format(year), doy='{}'.format(jday))
        st += st_read
        st.detrend('demean')
        st.taper(0.05) # find a good value!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        # this stream will be safed for the plot, therefor the stream is downsamplet but not processed
        st_dec = st_read.copy()
        st_dec = st_dec.decimate(15)  # downsampling for plot only
        #st_dec.detrend('demean')
        st_d += st_dec
        #st_d.merge()
        
    except:
        print('pass {}'.format(jday))
    return(st, st_d)

# save numpy array --------------------------------------------------------------------------------

def save_npy(save_path, save_filename, nparray):
    
    '''
    save_path: path, where you save the array as string
    save_filename: filename for your array as string
    nparray: array which you would like to save as numpy array
    
    returns: NOTHING
    '''
    
    # create new directory if it does not exist
    if not os.path.exists('{}'.format(save_path)):
        os.makedirs('{}'.format(save_path))
                          
    # save numpy array
    np.save('{}/{}'.format(save_path, save_filename),nparray) # .npy
    
    print('{} done'.format(save_filename))
                          
    return
    
# save numpy array compressed --------------------------------------------------------------------------------

def save_npz(save_path, save_filename, freqs, Pxx, start_times, rms_list, rmes_list, pgv_list, pga_list):
    
    '''
    save_path: path, where you save the array as string
    save_filename: filename for your array as string
    nparray: array which you would like to save as numpy array
    
    returns: NOTHING
    '''
    
    # create new directory if it does not exist
    if not os.path.exists('{}'.format(save_path)):
        os.makedirs('{}'.format(save_path))
                          
    # save numpy array
    np.savez_compressed('{}/{}'.format(save_path, save_filename), freqs=freqs, Pxx=Pxx, start_times=start_times,
                        rms_list=rms_list, rmes_list=rmes_list, pgv_list=pgv_list, pga_list=pga_list) # .npz
    
    print('{} done'.format(save_filename))
                          
    return    
    
# ---------------------------------------------------------------------------------    
    
def RSAM(data, samp_rate, datas, freq, Nm, N):
    filtered_data = obspy.signal.filter.bandpass(data, freq[0], freq[1], samp_rate)
    filtered_data = abs(filtered_data[:Nm])
    datas.append(filtered_data.reshape(-1,N).mean(axis=-1)*1.e9)
    return(datas)
    
def DSAR(data, samp_rate, datas, freqs_names, freqs, Nm, N):
    # compute dsar
    data = scipy.integrate.cumtrapz(data, dx=1./100, initial=0) # vel to disp
    data -= np.mean(data) # detrend('mean')
    j = freqs_names.index('mf')
    mfd = obspy.signal.filter.bandpass(data, freqs[j][0], freqs[j][1], samp_rate)
    mfd = abs(mfd[:Nm])
    mfd = mfd.reshape(-1,N).mean(axis=-1)
    j = freqs_names.index('hf')
    hfd = obspy.signal.filter.bandpass(data, freqs[j][0], freqs[j][1], samp_rate)
    hfd = abs(hfd[:Nm])
    hfd = hfd.reshape(-1,N).mean(axis=-1)
    dsar = mfd/hfd
    datas.append(dsar)
    return(datas)
    
# creates a df for each trace and append this df to a daily df
def create_df(datas, ti, freqs_names, df):
    datas = np.array(datas)
    time = [(ti+j*600).datetime for j in range(datas.shape[1])]
    df_tr = pd.DataFrame(zip(*datas), columns=freqs_names, index=pd.Series(time))
    df = pd.concat([df, df_tr])
    return(df)
    
# main function..............................................................................
def freq_bands_taper(sta,year,jday):   
    ''' 
    calculate and store power in 10 min long time windows for different frequency bands
    sensor measured ground velocity
    freqs: list contains min and max frequency in Hz
    dsar: float represents displacement (integration of)'''
    
    freqs_names = ['rsam','mf','hf', 'dsar']
    df = pd.DataFrame(columns=freqs_names)
    daysec = 24*3600
    freqs = [[2, 5], [4.5, 8], [8,16]]
    
    st, st_dec = read_taper_stream(sta,year,jday)

    if len(st)>0: # if stream not empty
        for tr in st:
            datas = []
            data = tr.data
            samp_rate = tr.meta['sampling_rate']
            ti = tr.meta['starttime']
            # round start time to nearest 10 min increment
            tiday = obspy.UTCDateTime("{:d}-{:02d}-{:02d} 00:00:00".format(ti.year, ti.month, ti.day)) # date
            ti = tiday+int(np.round((ti-tiday)/600))*600 # nearest 10 min to starttime
            N = int(600*samp_rate)    # 10 minute windows in seconds
            Nm = int(N*np.floor(len(data)/N)) # np.floor rounds always to the smaller number
            # seconds per day (86400) * sampling rate (100) -> datapoints per day

            for freq, frequ_name in zip(freqs, freqs_names[:3]):
                datas = RSAM(data, samp_rate, datas, freq, Nm, N) # get RSAM for different frequency bands

            datas = DSAR(data, samp_rate, datas, freqs_names, freqs, Nm, N)
        
            df = create_df(datas, ti, freqs_names, df)
        #print(df)   
        
        
        
        if not os.path.exists('tmp_{}/{}'.format(year, sta)):
            os.makedirs('tmp_{}/{}'.format(year, sta))
        
        df.to_csv('tmp_{}/{}/_tmp_taper_{}_{}.csv'.format(year,sta,sta,jday), index=True, index_label='time')
        
    return(st_dec)
