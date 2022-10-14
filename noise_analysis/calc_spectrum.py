# impor the modules and path
import sys
sys.path.append("/data/wsd01/pnwstore/")

import numpy as np
import pandas as pd
import obspy
import obspy.signal.filter
from obspy import UTCDateTime
from pnwstore.mseed import WaveformClient
client = WaveformClient()
import datetime
import scipy
import glob
import sys
import os
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import multiprocessing
from functools import partial

sys.path.append('/home/koepflma/project1/Mt-St-Helens')
from functions import *

# set parameters --------------------------------------------------------------------------------------------------------------

net = 'UW'
sta = 'EDM'
cha = 'EHZ'
year = 2018
jday_range = np.arange(1,10+1)
hour = 12
yaxis = 'log' # log or lin

def main(jday, net, sta, cha, year, hour, yaxis):
    
    # read data to stream -----------------------------------------------------------------------------------------------------
    try:
        st_r = read_stream(net, sta, cha, year, jday)
    except:
        pass

    # correct insrument response -----------------------------------------------------------------------------------------------
    inv = obspy.read_inventory('/auto/pnwstore1-wd11/PNWStationXML/{}/{}.{}.xml'.format(net,net,sta))
    pre_filt = [1e-3, 5e-2, 45, 50]
    water_level = 60
    st = st_r
    for tr in st:
        s_time_str = str(tr.stats['starttime']).split('.')[0].replace(':', '-')
        tr.remove_response(inventory=inv, zero_mean=True,taper=True, taper_fraction=0.05,
                              pre_filt=pre_filt, output="VEL", water_level=water_level,
                              plot=False)
    #                           plot='sensor_response_tests/{}__pre_filt{}-{}_{}-{}__water_level{}.png'.format(
    #                           s_time_str,
    #                           pre_filt[0], pre_filt[1], pre_filt[2], pre_filt[3], water_level))
    
    
    # merge traces ------------------------------------------------------------------------------------------------------------
    st = st_r
    st.merge()
    
    # create spectrogram ------------------------------------------------------------------------------------------------------
    starttime = UTCDateTime(year=year, julday=jday, hour=hour).datetime
    date = starttime.date()
    t0 = pd.to_datetime(starttime) # equal to start (defines starttime for gliding lines)
    t0 = t0.value
    start=obspy.UTCDateTime(starttime) # start time for spectrogram, qual to t0
    #timeslot=int(end-start) # in sec
    timeslots = [60,60*15,60*30,60*60,60*60*2,60*60*3,60*60*4,60*60*5,60*60*6] # in sec


    fig, ax = plt.subplots(figsize=(7,5))

    for timeslot in timeslots:

        #df_cut = df_roll.loc[str(date)]
        #df_cut = df_cut[:27800]
        st = st.merge()
        tr = st[0] #only get one obspy trace from obspy stream
        trslice = tr.slice(tr.stats.starttime+0, tr.stats.starttime+timeslot)
        Pxx, freqs = matplotlib.mlab.psd(trslice.data, NFFT=int(tr.stats.sampling_rate*10), Fs=trslice.stats.sampling_rate)
        if timeslot == 600:
            ax.plot(freqs, Pxx, label='{} s'.format(timeslot),linestyle='-',color='k',linewidth=2)
        else:
            ax.plot(freqs, Pxx, label='{} s'.format(timeslot),linestyle='--',alpha=0.5)

    for n in np.arange(2.1,10,1):
        x = n**(3/2)
    #    ax.vlines(x,-1,1,color='gray')#,linestyles='dotted')
    ax.set_xlim(0.5,50)
    ax.set_ylim(1e-20,15e-13)
    ax.legend(ncol=2)
    ax.set_title('Frequency spectrum {}'.format(date),size=18)
    ax.set_xlabel('Frequency [Hz]',size=12)
    #plt.gca().axes.get_yaxis().set_visible(False)
    ax.set_ylabel('PSD [m²/s²/Hz]',size=12)
    ax.grid()
    
    if yaxis == 'log':
        plt.yscale('log')
    # save figure ------------------------------------------------------------------------------------------------------------
        save_filename = '{}_{}_{}_log.png'.format(year, jday, sta) # file name
   
    if yaxis == 'lin':
        save_filename = '{}_{}_{}.png'.format(year, jday, sta) # file name

    save_path = 'spectrum/{}/{}/'.format(year,sta) # path where to save file
        
    if not os.path.exists(save_path): # create folders from save_path if not exists
        os.makedirs(save_path)
    
    fig.savefig(save_path+save_filename, dpi=300, bbox_inches='tight')
    #plt.show()

    return

# multiprocessing ------------------------------------------------------------------------------------------------------------
p = multiprocessing.Pool(processes=10)
p.map(partial(main, net=net, sta=sta, cha=cha,  year=year, hour=hour, yaxis=yaxis), jday_range)
p.close()
p.join()