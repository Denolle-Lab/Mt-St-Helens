'''
Script to  visualize the number of correlation functions that have passed
the quality control depnending on time.
'''

import os
import numpy as np
from matplotlib import pyplot as plt
from obspy import UTCDateTime

from seismic.monitor.dv import read_dv
from seismic.plot.plot_utils import set_mpl_params

# Folder with the dvs
infiles = '/data/wsd01/st_helens_peter/dv/new_gap_handling/*interp_{freq}-*srw/*.npz'
infiles2 = '/data/wsd01/st_helens_peter/dv/dv_separately/xstations_{freq}-*/*.npz'

outfile = '/data/wsd01/st_helens_peter/dv/new_gap_handling/availability_{freq}.png'
outnpz = '/data/wsd01/st_helens_peter/dv/new_gap_handling/availability_{freq}.npz'


set_mpl_params()
for freq in 0.25*2**np.arange(3):
    dvs = read_dv(infiles.format(freq=freq))
    dvs += read_dv(infiles2.format(freq=freq))
    t = [dv.stats.corr_start for dv in dvs]
    # find time series with maximum length
    t = t[np.argmax([len(t_) for t_ in t])]
    # convert to datetime
    t = np.array([t_.datetime for t_ in t])
    # get number of available correlation functions
    n_avail = np.zeros_like(t)
    for i, t_ in enumerate(t):
        dv_sel = [dv for dv in dvs if (min(dv.stats.corr_start) <= UTCDateTime(t_) and max(dv.stats.corr_start) >= UTCDateTime(t_))]
        ii = [np.argmin(abs(np.array(dv.stats.corr_start) - UTCDateTime(t_))) for dv in dv_sel]
        n_avail[i] = np.sum(np.array([dv.avail[ii_] for ii_, dv in zip(ii, dv_sel)]).astype(int))
    np.savez(outnpz.format(freq=freq), t=t, n=n_avail)
    plt.figure(figsize=(12,9))
    
    plt.fill_between(t, n_avail, color='g')
    plt.ylabel(r'$N_{CF}$')
    plt.ylim((0, None))
    plt.xlim((min(t), max(t)))
    plt.title(f'available CF {freq}-{freq*2} Hz')
    plt.savefig(outfile.format(freq=freq), dpi=300, facecolor='None')
    plt.close()
