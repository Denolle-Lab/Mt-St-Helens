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
import tsfel
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
    
# creates a df for each trace and append this df to a daily df
# def create_df(datas, ti, freqs_names, df):
#     datas = np.array(datas)
#     time = [(ti+j*600).datetime for j in range(datas.shape[1])]
#     df_tr = pd.DataFrame(zip(*datas), columns=freqs_names, index=pd.Series(time))
#     df = pd.concat([df, df_tr])
#     return(df)    

def create_df(datas, ti, freqs_names, df):
    datas = np.array(datas)
    time = [(ti+j*600).datetime for j in range(datas.shape[1])]
    df_tr = pd.DataFrame(zip(*datas), columns=freqs_names+['rms','rmes','pgv','pga'], index=pd.Series(time))
    df = pd.concat([df, df_tr])
    return(df) 
    
# main function..............................................................................
def tsfel_calc(jday, year, netstacha):   
    ''' 
    calculate and store power in 10 min long time windows for different frequency bands
    sensor measured ground velocity
    freqs: list contains min and max frequency in Hz
    dsar: float represents displacement (integration of)'''
    
    net = netstacha.split('-')[0]
    sta = netstacha.split('-')[1]
    cha = netstacha.split('-')[2]
    
#     file_path = '/data/wsd03/data_manuela/MtStHelens/RSAM_DSAR/tmp_{}/{}/'.format(year, sta)
    file_path = './tmp_{}/{}/'.format(year, sta)
    file_name = '{}_{}.csv'.format(sta,jday)
        
    if os.path.isfile(file_path+file_name):
        print('file for {}-{} at {} already exist'.format(year,jday, netstacha))
        pass
    else:    
        start_time = time.time()
        daysec = 24*3600

        st = preprocessing(year,jday, net, sta, cha)
        print(st)

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

#                 datas = DSAR(data, samp_rate, datas, freqs_names, freqs, Nm, N)
                cfg = tsfel.get_features_by_domain()
                X = tsfel.time_series_features_extractor(cfg, data)

#                 df = create_df(datas, ti, freqs_names, df)
            
            if not os.path.exists(file_path):
                os.makedirs(file_path)
                
            X.to_csv(file_path + file_name, index=True, index_label='time')
            print('One day tooks {} seconds.'.format(round(time.time()-start_time),3))
        else:
            print('empty stream station {} day {}'.format(sta,jday))
    return()

year = 2004
jday = 100

# stations as string 'network-station-channel'
s1  = 'UW-EDM-EHZ'
s2  = 'UW-SHW-EHZ'
s3  = 'UW-HSR-EHZ'
s4  = 'UW-SOS-EHZ'
s5  = 'UW-JUN-EHZ'
s6  = 'UW-ELK-EHZ'
s7  = 'UW-TDL-EHZ'
s8  = 'UW-SUG-EHZ'
s9  = 'UW-YEL-EHZ'
s10 = 'UW-FL2-EHZ'
s11 = 'UW-CDF-?HZ' #'UW-CDF-?H?' # eighter HHZ or EHZ

s12 = 'UW-SEP-?HZ' #'UW-SEP-?H?'
s13 = 'CC-SEP-?HZ' #'CC-SEP-?H?' # only one of the two SEP at the time & either EHZ or BHZ
s14 = 'UW-STD-EHZ'
s15 = 'CC-STD-BHZ'

s16 = 'CC-VALT-BHZ' #'CC-VALT-BH?'
s17 = 'CC-JRO-BHZ'
s18 = 'CC-HOA-BHZ' #'CC-HOA-BH?'
s19 = 'CC-LOO-BHZ' #'CC-LOO-BH?'
s20 = 'CC-USFR-BHZ' #'CC-USFR-BH?'
s21 = 'CC-NED-EHZ'
s22 = 'CC-REM-BHZ' #'CC-REM-BH?'
s23 = 'CC-SWFL-BHZ' #'CC-SWFL-BH?'
s24 = 'CC-SFW2-BHZ' #'CC-SFW2-BH?'
s25 = 'CC-MIDE-EHZ'
s26 = 'CC-MIBL-EHZ'
s27 = 'CC-BLIS-EHZ'
s28 = 'CC-RAFT-EHZ'
s29 = 'CC-SPN5-EHZ'
s30 = 'CC-SEND-EHZ'

list_stations = [s1]#[s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,
                 #s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,s25,s26,s27,s28,s29,s30] # make a list of all stations
    
    
for netstacha in list_stations:
    tsfel_calc(jday, year, netstacha)