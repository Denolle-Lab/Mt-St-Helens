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
sta = 'SHW'
cha = 'EHZ'
year = 2018
jday_range = np.arange(100,110+1)

def main(jday, net, sta, cha, year):
    
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
    st.merge()
    
    # create spectrogram ------------------------------------------------------------------------------------------------------
    tr = st[0] # only get one obspy trace from obspy stream
    data = tr.data # extract numpy data array from obspy trace
    fs = tr.stats.sampling_rate #sampling rate

    NFFT = fs * 100 # bin size for fourier transform. Type in length (seconds)

    fig = plt.figure(figsize=(6.4*2,4.8)) #create figure and add axes to it
    # ax1 = fig.add_axes([0.1, 0.1, 0.7, 0.60]) #[left bottom width height]
    # ax2 = fig.add_axes([0.83, 0.1, 0.03, 0.6])
    ax1 = fig.add_axes([0.125, 0.125, 0.76, 0.6])
    ax2 = fig.add_axes([0.895, 0.125, 0.02, 0.6]) # colorbar

    #plot spectrogram on first axis
    Pxx, freqs, bins, im = ax1.specgram(data, NFFT=int(NFFT), Fs=fs,
                                        noverlap=0,         # overlap of bins in samples
                                        detrend='linear',   # detrending before taking fourier transform
                                        mode='psd',         # 'psd', 'magnitude', 'angle', 'phase'
                                        scale_by_freq=True, # unit/Hz
                                        scale='dB',         #'linear', 'dB'
                                        cmap='viridis',     # your favourite colormap
                                        vmin=-200,
                                        vmax=-50,
                                       )

    # show hour of day on x-axis
    h_int = 6
    x_ticks = np.arange(0,24+h_int,h_int)
    x_tickloc = np.linspace(np.min(bins), np.max(bins), len(x_ticks))
    ax1.set_xticks(x_tickloc)
    ax1.set_xticklabels(['{}/{}  {:02d}:00'.format(tr.stats.starttime.month,tr.stats.starttime.day,
                                                   x) for x in x_ticks])
    ax1.set_xlim(np.min(bins), np.max(bins))

    ax1.set_xlabel('UTC [Month/Day  Hour:Minute]', fontsize=12) #x-label
    ax1.set_ylabel('Frequency [Hz]', fontsize=12) #y-label
    ax1.tick_params(axis='both', labelsize=12)

    ax1.set_yscale('log')
    ax1.set_ylim([0.01,50]) #be carefull with lower limit when y-scale is logarithmic

    cbar = plt.colorbar(im, cax=ax2) #map colorbar to image (output of specgram), plot it on ax2
    cbar.set_label('Power Spectral Density [dB]', fontsize=12) #colorbar label
    cbar.ax.locator_params(nbins=5)
    
    # save figure ------------------------------------------------------------------------------------------------------------
    
    save_path = 'spectrogram/{}/{}/'.format(year,sta) # path where to save file
    save_filename = '{}_{}_{}.png'.format(year, jday, sta) # file name

    if not os.path.exists(save_path): # create folders from save_path if not exists
        os.makedirs(save_path)
    
    fig.savefig(save_path+save_filename, dpi=300, bbox_inches='tight')
    #plt.show()

    return

# multiprocessing ------------------------------------------------------------------------------------------------------------
p = multiprocessing.Pool(processes=10)
p.map(partial(main, net=net, sta=sta, cha=cha,  year=year), jday_range)
p.close()
p.join()