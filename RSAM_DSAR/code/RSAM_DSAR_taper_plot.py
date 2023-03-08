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

year = 2006
trim = False

st_r = obspy.read('tmp_{}/st_{}.mseed'.format(year,year)) # read obspy stream

# crate df with index time and columns rsam, mf, hf, dsar
all_files = sorted(glob.glob('tmp_{}/_tmp_taper_*.csv'.format(year,year)))
#all_files = all_files[1:]
li = []
for filename in all_files:
    frame = pd.read_csv(filename)
    li.append(frame)

df = pd.concat(li, axis=0, ignore_index=True)
df.set_index('time',inplace=True)
df.index = pd.to_datetime(df.index).tz_localize(None)

if trim==True: # trim stream ---------------------------------------------------------------------------
    start_trim = obspy.UTCDateTime(2004,4,1)
    end_trim = obspy.UTCDateTime(2004,5,1)
    st_trim = st_r.copy()
    st_trim.trim(start_trim, end_trim)
    st_trim.merge()
    
    fig, ax = plt.subplots(2,1, sharex=True, figsize=(6.4*2, 4.8))
    ax[0].plot(st_trim[0].times('matplotlib'), st_trim[0].data, color='k', 
               label='{}.{}..{}'.format(st_trim[0].stats['network'],
                                        st_trim[0].stats['station'],
                                        st_trim[0].stats['channel']))
    ax[1].plot(df['rsam'], label='LF')
    ax[1].plot(df['mf'], label='MF')
    ax[1].plot(df['hf'], label='HF')
    ax[1].plot(np.nan, label='DSAR')

    ax2 = ax[1].twinx()
    ax2.plot(df['dsar'], label='DSAR', color='C3')
    #ax2.set_ylim(0,2.5)
    ax[1].set_xlim(start_trim.datetime, end_trim.datetime)

    ax[1].set_ylabel('RSAM')
    ax[1].set_ylim(0,2e11)
    ax2.set_ylabel('DSAR')
    ax[0].legend(loc='upper left')
    ax[1].legend(ncol=4, loc='upper left')

    fig.savefig('plots/{}_EDM_1m_taper.png'.format(year), bbox_inches='tight', dpi=300)
    
if trim==False: # plot oe year -------------------------------------------------------------------------
    st_long = st_r.copy()
    st_long.merge()
    
    fig, ax = plt.subplots(2,1, sharex=True, figsize=(6.4*2, 4.8))
    ax[0].plot(st_long[0].times('matplotlib'), st_long[0].data, color='k', 
               label='{}.{}..{}'.format(st_long[0].stats['network'],st_long[0].stats['station'],st_long[0].stats['channel']))
    #ax[0].set_ylim(-4000,4000)
    ax[1].plot(df['rsam'], label='RSAM')
    ax[1].plot(df['mf'], label='MF')
    ax[1].plot(df['hf'], label='HF')
    ax[1].plot(np.nan, label='DSAR')

    ax2 = ax[1].twinx()
    ax2.plot(df['dsar'], label='DSAR', color='C3')
    #ax2.set_ylim(0,2.5)

    ax[0].legend(loc='upper left')
    ax[1].legend(ncol=4, loc='upper left')
    fig.savefig('plots/{}_EDM_1y_taper.png'.format(year), bbox_inches='tight', dpi=300)

