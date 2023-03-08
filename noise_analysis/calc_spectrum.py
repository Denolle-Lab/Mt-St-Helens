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

# net = 'UW'
# sta = 'SHW'
# cha = 'EHZ'

s1 = 'UW-EDM-EHZ' # station 1
s2 = 'UW-SHW-EHZ' # station 2
s3 = 'UW-HSR-EHZ' # station 3
s4 = 'CC-JRO-BHZ' # station 4
s5 = 'UW-SOS-EHZ' # station 5
s6 = 'CC-VALT-BHZ' # station 6
s7 = 'CC-SEP-EHZ' # station 7
s8 = 'CC-STD-BHZ' # station 8
s9 = 'UW-JUN-EHZ' # station 9
s_list = [s1,s2,s3,s4,s5,s6,s6,s7,s8,s9]

year = 2004
jday_range = np.arange(1,5+1)
hour = 12

day_netstacha = [tuple([str(i)]+[s]) for i in jday_range for s in s_list]

axis = 'log' # log or lin (for x and y axis)

def main(day_netstacha, year, hour, axis):
    
    jday = int(day_netstacha[0]) # julian day
    netstacha = day_netstacha[1].split('-') # station code for example: 'UW-EDM-EHZ'
    net = netstacha[0] # network
    sta = netstacha[1] # station
    cha = netstacha[2] # channel
    
    # read data to stream -----------------------------------------------------------------------------------------------------
    try:
        st_r = read_stream(net, sta, cha, year, jday)


        # correct insrument response ------------------------------------------------------------------------------------------
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


        # merge traces -------------------------------------------------------------------------------------------------------
        st.merge()

        # create spectrogram -------------------------------------------------------------------------------------------------
        starttime = UTCDateTime(year=year, julday=jday, hour=hour).datetime
        date = starttime.date()
        t0 = pd.to_datetime(starttime) # equal to start (defines starttime for gliding lines)
        t0 = t0.value
        start=obspy.UTCDateTime(starttime) # start time for spectrogram, qual to t0
        #timeslot=int(end-start) # in sec
        timeslots = [60,60*15,60*30,60*60,60*60*2,60*60*3,60*60*4,60*60*5,60*60*6] # in sec


        fig, ax = plt.subplots(figsize=(7,5))

        for timeslot in timeslots:

            tr = st[0] #only get one obspy trace from obspy stream
            trslice = tr.slice(tr.stats.starttime+0, tr.stats.starttime+timeslot)
            Pxx, freqs = matplotlib.mlab.psd(trslice.data, NFFT=int(tr.stats.sampling_rate*10), Fs=trslice.stats.sampling_rate)
            if timeslot == 600:
                ax.plot(freqs, Pxx, label='{} s'.format(timeslot),linestyle='-',color='k',linewidth=2)
            else:
                ax.plot(freqs, Pxx, label='{} s'.format(timeslot),linestyle='--',alpha=0.5)
        # overtones ---------------------------------------------------------------------------------------------------------
#         for n in np.arange(2.1,10,1):
#             x = n**(3/2)
#             ax.vlines(x,-1,1,color='gray')#,linestyles='dotted')
        ax.set_xlim(0.5,50)
        ax.set_ylim(1e-20,15e-13)
        ax.legend(ncol=2)
        ax.set_title('Frequency spectrum {}'.format(date),size=18)
        ax.set_xlabel('Frequency [Hz]',size=12)
        #plt.gca().axes.get_yaxis().set_visible(False)
        ax.set_ylabel('PSD [m²/s²/Hz]',size=12)
        ax.grid()

        if axis == 'log':
            plt.xscale('log')
            plt.yscale('log')
        # save figure -------------------------------------------------------------------------------------------------------
            save_filename = '{}_{}_{}_log.png'.format(year, str(jday).zfill(3), sta) # file name

        if axis == 'lin':
            save_filename = '{}_{}_{}.png'.format(year, str(jday).zfill(3), sta) # file name

        save_path = 'spectrum/{}/{}/'.format(year,sta) # path where to save file

        if not os.path.exists(save_path): # create folders from save_path if not exists
            os.makedirs(save_path)

        fig.savefig(save_path+save_filename, dpi=300, bbox_inches='tight')
        #plt.show()

    except:
        print(day_netstacha)
        pass
    
    return

# multiprocessing ------------------------------------------------------------------------------------------------------------
p = multiprocessing.Pool(processes=10)
p.map(partial(main, year=year, hour=hour, axis=axis), day_netstacha)
p.close()
p.join()