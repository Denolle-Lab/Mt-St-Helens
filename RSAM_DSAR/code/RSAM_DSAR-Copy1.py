import numpy as np
import pandas as pd
import obspy
import obspy.signal.filter
import datetime
import scipy
import glob
import sys
import os
import scipy as sc
import time
import matplotlib.pyplot as plt
import argparse
import warnings
warnings.filterwarnings("ignore")

import multiprocessing
from functools import partial

sys.path.append("/data/wsd01/pnwstore/")
from pnwstore.mseed import WaveformClient
client = WaveformClient()

# Define all functions---------------------------------------------------

def preprocessing(year,jday, net, sta, cha):

    try:
        # this stream will be used for RSAM and DSAR calculations
        #st = obspy.read('/1-fnp/pnwstore1/p-wd05/PNW2004/UW/2004/{}/EDM.UW.2004.{}'.format(jday,jday))
        st = client.get_waveforms(network=net, station=sta, channel=cha,
                                       year='{}'.format(year), doy='{}'.format(jday))

        st.detrend('linear')
        st.taper(max_percentage=None,max_length=5, type='hann') #max_length in sec
        
        # correct insrument response
        inv = obspy.read_inventory('/auto/pnwstore1-wd11/PNWStationXML/{}/{}.{}.xml'.format(net,net,sta))
        pre_filt = [1e-3, 5e-2, 45, 50]
        water_level = 60
        
        for tr in st:
            tr.remove_response(inventory=inv, zero_mean=True,taper=True, taper_fraction=0.05,
                                      pre_filt=pre_filt, output="VEL", water_level=water_level,
                                      plot=False)

            # correct positive dip
            dip = inv.get_orientation(tr.id, datetime=tr.stats.starttime)['dip']
            if dip > 0:
                tr.data *= -1
#         st.merge(fill_value=0)
        print(':) year={}, jday={}, net={}, sta={}, cha={}'.format(year,jday, net, sta, cha))
    except:
        print('pass station {} day {}'.format(sta,jday))
    return(st)
    
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

def nDSAR(datas):
    dsar = datas[3]
    ndsar = dsar/sc.stats.zscore(dsar)
    datas.append(ndsar)
    return(datas)
    
# creates a df for each trace and append this df to a daily df
def create_df(datas, ti, freqs_names, df):
    datas = np.array(datas)
    time = [(ti+j*600).datetime for j in range(datas.shape[1])]
    df_tr = pd.DataFrame(zip(*datas), columns=freqs_names, index=pd.Series(time))
    df = pd.concat([df, df_tr])
    return(df)    
    
# main function..............................................................................
def freq_bands_taper(jday, year, net, sta, cha):   
    ''' 
    calculate and store power in 10 min long time windows for different frequency bands
    sensor measured ground velocity
    freqs: list contains min and max frequency in Hz
    dsar: float represents displacement (integration of)'''
    
    start_time = time.time()
    freqs_names = ['rsam','mf','hf','dsar']
    df = pd.DataFrame(columns=freqs_names)
    daysec = 24*3600
    freqs = [[2,5], [4.5,8], [8,16]]
    
    st = preprocessing(year,jday, net, sta, cha)

    if len(st)>0: # if stream not empty
#         st.resample(50)
        for tr in st:
#         tr = st[0]
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
#             datas = nDSAR(datas) # --> add ndsar in freqs_names

#             noise_analysis(data, N, Nm, samp_rate)

            df = create_df(datas, ti, freqs_names, df)
        
        if not os.path.exists('/data/wsd03/data_manuela/MtStHelens/RSAM_DSAR/{}/{}'.format(year, sta)):
            os.makedirs('/data/wsd03/data_manuela/MtStHelens/RSAM_DSAR/{}/{}'.format(year, sta))
        df.to_csv('/data/wsd03/data_manuela/MtStHelens/RSAM_DSAR/{}/{}/{}_{}.csv'.format(year,sta,sta,jday), index=True, index_label='time')
        print('One day tooks {} seconds.'.format(round(time.time()-start_time),3))
    return()


# end define functions------------------------------------------------------------------------

# parser = argparse.ArgumentParser(description='Calculate different frequency bands of RSMA and DSAR.')
# parser.add_argument('year', type=int, help='Year of interest')
# parser.add_argument('start_day', type=int, help='Julian day you want to start')
# parser.add_argument('end_day', type=int, help='Julian day +1 you want to end')
# parser.add_argument('net', type=str, help='Network you want to process')
# parser.add_argument('sta', type=str, help='Station you want to process')
# parser.add_argument('cha', type=str, help='Channel you want to process')

# args = parser.parse_args()

# year = args.year
# jdays = ['{:03d}'.format(jday) for jday in range(args.start_day,args.end_day)]
# net = args.net
# sta = args.sta
# cha = args.cha

#--> python RSAM_DSAR.py 2004 2 3 'UW' 'EDM' 'EHZ'

# calculate frequencie bands AND save stream
# single processing write downsampled stream -----------------------------------------------------------------------------

#st_long = obspy.Stream()
#for i, jday in enumerate(jdays,1):
#    st_dec = freq_bands_taper(sta,year,jday)
#    st_long += st_dec
    
#    sys.stdout.write('\r{} of {}\n'.format(i, len(jdays)))
#    sys.stdout.flush()

# multi processing write downsampled stream -----------------------------------------------------------------------------
#st_long.write("tmp_{}/st_{}_{}.mseed".format(year,sta,year), format="MSEED") # save stream

# # st_long = obspy.Stream()
# # for i, st_d in enumerate(p.imap(partial(freq_bands_taper,sta,year),jdays),1):
    
# #     st_long += st_d # st is downsampled
    
# #     sys.stdout.write('\r{} of {}'.format(i, len(jdays)))
# #     sys.stdout.flush()
#st_long.write("tmp_{}/{}/st_{}_{}.mseed".format(year,sta,sta,year), format="MSEED") # save stream

stime = time.time()
p = multiprocessing.Pool(processes=24)
p.imap_unordered(partial(freq_bands_taper,year=year, net=net, sta=sta, cha=cha), jdays)
p.close()
p.join()

# multi processing for several station -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Calculate different frequency bands of RSMA and DSAR.')
parser.add_argument('year', type=int, help='Year of interest')
parser.add_argument('start_day', type=int, help='Julian day you want to start')
parser.add_argument('end_day', type=int, help='Julian day +1 you want to end')
parser.add_argument('net', type=str, help='Network you want to process')
parser.add_argument('cha', type=str, help='Channel you want to process')

args = parser.parse_args()

year = args.year
jdays = ['{:03d}'.format(jday) for jday in range(args.start_day,args.end_day)]
net = args.net
cha = args.cha

for sta in ['EDM','SHW','HSR','SOS','TDL','ELK','FL2','CDF','JUN','YEL','SEP','STD']:
    print('Station {}'.format(sta))
    stime = time.time()
    p = multiprocessing.Pool(processes=24)
    p.imap_unordered(partial(freq_bands_taper,year=year, net=net, sta=sta, cha=cha), jdays)
    p.close()
    p.join()
    print('Calculation tooks {} seconds.'.format(round(time.time()-stime),3))

# comment parser out
#--> python RSAM_DSAR.py 2004 1 366 'UW' 'EHZ'


# call function to test-----------------------------------------------------------------------------
# freq_bands_taper(2004,'002','UW','EDM','EHZ')
