import numpy as np
import pandas as pd
import obspy
import obspy.signal.filter
import datetime
import scipy
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

# Define all functions---------------------------------------------------

def read_stream(sta,year,jday):
    st = obspy.Stream()
    st_d = obspy.Stream()
    try:
        # this stream will be used for RSAM and DSAR calculations
        #st_read = obspy.read('/1-fnp/pnwstore1/p-wd05/PNW2004/UW/2004/{}/EDM.UW.2004.{}'.format(jday,jday))
        st_read = client.get_waveforms(network='UW', station='{}'.format(sta), channel='*', year='{}'.format(year), doy='{}'.format(jday))
        st += st_read
        st.detrend('demean')
        st.merge(method=0, fill_value=0, interpolation_samples=0)
        
        # this stream will be safed for the plot, therefor the stream is downsamplet but not processed
        st_dec = st_read.copy()
        st_dec = st_dec.decimate(15)  # downsampling for plot only
        #st_dec.detrend('demean')
        st_d += st_dec
        #st_d.merge()
        
    except:
        print('pass {}'.format(jday))
    return(st, st_d)
    
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
    
    st, st_dec = read_stream(sta,year,jday)

    if len(st)>0: # if stream not empty
        tr = st[0]
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
        
        if not os.path.exists('tmp_{}/{}'.format(year, sta)):
            os.makedirs('tmp_{}/{}'.format(year, sta))
        df.to_csv('tmp_{}/{}/_tmp_fl_{}_{}.csv'.format(year,sta,sta,jday), index=True, index_label='time')
    return(st_dec)


# end define functions------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Calculate different frequency bands of RSMA and DSAR.')
parser.add_argument('sta', type=str, help='Station you want to process')
parser.add_argument('year', type=int, help='Year of interest')
parser.add_argument('start_day', type=int, help='Julian day you want to start')
parser.add_argument('end_day', type=int, help='Julian day +1 you want to end')
args = parser.parse_args()

sta = args.sta
year = args.year
jdays = ['{:03d}'.format(jday) for jday in range(args.start_day,args.end_day)]

## python RSAM_DSAR_taper.py 2004 2 3

# calculate frequencie bands AND save stream
# single processing -----------------------------------------------------------------------------

#st_long = obspy.Stream()
#for i, jday in enumerate(jdays,1):
#    st_dec = freq_bands_taper(sta,year,jday)
#    st_long += st_dec
    
#    sys.stdout.write('\r{} of {}\n'.format(i, len(jdays)))
#    sys.stdout.flush()

#st_long.write("tmp_{}/st_{}_{}.mseed".format(year,sta,year), format="MSEED") # save stream

# multi processing -----------------------------------------------------------------------------
import multiprocessing
from functools import partial

p = multiprocessing.Pool(processes=16)
st_long = obspy.Stream()
for i, st_d in enumerate(p.imap(partial(freq_bands_taper,sta,year),jdays),1):
    
    st_long += st_d # st is downsampled
    
    sys.stdout.write('\r{} of {}'.format(i, len(jdays)))
    sys.stdout.flush()
p.close()
p.join()

#st_long.write("tmp_{}/{}/st_{}_{}.mseed".format(year,sta,sta,year), format="MSEED") # save stream



