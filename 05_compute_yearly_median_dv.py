'''
Computes the annual median of the dv/v time series corresponding
to channel combinations.
As in Makus et al. (2024, SRL)

:copyright:
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 26th February 2024 02:05:57 pm
Last Modified: Monday, 26th February 2024 02:08:38 pm
'''

import glob
import os

from seismic.monitor.dv import read_dv
import numpy as np
from obspy import UTCDateTime
from obspy.signal.filter import highpass
from matplotlib import pyplot as plt


proj_dir = '/your/project/directory/'

freqs = [0.25, 0.5, 1.0]

for freq0 in freqs:
    infiles = glob.glob(os.path.join(
        proj_dir, 
        f'dv/*_{freq0}-{freq0*2}*/*.npz'))
    infiles += glob.glob(os.path.join(
        proj_dir,
        f'dv/dv_separately/xstations_{freq0}-{freq0*2}_*/*.npz'))
    dv_medians = []
    for infile in infiles:
        dv = read_dv(infile)
        # plt.figure()
        nani = np.isnan(dv.corr)

        low_corr = dv.corr < .35

        offset_1d = np.nan_to_num(dv.value, copy=True)
        df = 1/(dv.stats.corr_start[1]-dv.stats.corr_start[0])
        filtfreq = 1/(365.25*24*60*60*2)
        # mask nans
        val_hpf = highpass(offset_1d, filtfreq, df, corners=4, zerophase=True)
        val_hpf[nani] += np.nan
        val_hpf[low_corr] += np.nan

        years = np.array([_date.year for _date in dv.stats.corr_start])
        doy = np.array([UTCDateTime(_date).julday for _date in dv.stats.corr_start])
        years_unique = np.unique(years)

        x_data = []
        y_data = []
        
        for year in years_unique:
            # plt.plot(doy[years==year], dv_hpf[freq][m, n, years==year], label=year, alpha=.2, color='k')
            # collect data to compute median
            x_data.append(doy[years==year])
            y_data.append(val_hpf[years==year])
        # intepolate all these curves for the same doys
        xq = np.arange(1, 366)
        yq = np.zeros((len(years_unique), len(xq)))
        for ii, (x_datum, y_datum) in enumerate(zip(x_data, y_data)):
            yq[ii] = \
                np.interp(xq, x_datum, y_datum)
            # demean
            yq[ii] -= np.nanmean(yq[ii])
        y_median = np.nanmedian(yq, axis=0)
        dv_medians.append(y_median)
    dv_medians = np.array(dv_medians)*-100
    plt.plot(xq, dv_medians.T, alpha=.3, color='k');
    plt.plot(xq, np.nanmedian(dv_medians, axis=0), color='r', linewidth=2)
    plt.ylabel('dv/v [%]')
    plt.xlabel('day of year')
    outfolder = os.path.join(
        proj_dir, 'figures', 'dv_median'))
    os.makedirs(outfolder, exist_ok=True)
    plt.savefig(os.path.join(outfolder, f'{freq0}_all.png'), dpi=300)
    plt.close()
    plt.figure()
    plt.plot(xq, np.nanmedian(dv_medians, axis=0), color='r', linewidth=2)
    plt.ylabel('dv/v [%]')
    plt.xlabel('day of year')
    plt.savefig(os.path.join(outfolder, f'{freq0}_median.png'), dpi=300)
    plt.close()
    # save the actual data
    np.savez(os.path.join(outfolder, f'{freq0}_medians.npz'), medians=dv_medians)
