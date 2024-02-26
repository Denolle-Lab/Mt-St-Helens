'''
Plots CFs and dv/v in one figure as in Makus et al, 2024 (SRL).

:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 15th February 2024 03:36:05 pm
Last Modified: Monday, 26th February 2024 10:20:33 am
'''

import os
import glob
import locale

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
import matplotlib.dates as mdates

from seismic.db.corr_hdf5 import CorrelationDataBase as CDB
from seismic.monitor.dv import read_dv
from seismic.plot.plot_utils import set_mpl_params

outfolder = '../../figures/paper/CF_dv_merge/selfcorrs'
indir = '/data/wsd01/st_helens_peter/corrs_response_removed/betweenComponents_5_0.5-1.0_wl120.0_1b_SW/'
dvindir = '/data/wsd01/st_helens_peter/dv/new_gap_handling/betweenComponents_td_taper_no_gap_interp_0.5-1.0_wl432000_tw8.0-100.0__1b_mute_SW_presmooth30d_srw'

# outfolder = '../../figures/paper/CF_dv_merge/autocorrs'
# indir = '/data/wsd01/st_helens_peter/corrs_response_removed/autoComponents_5_0.5-1.0_wl120.0_1b/'
# dvindir = '/data/wsd01/st_helens_peter/dv/new_gap_handling/autoComponents_td_taper_no_gap_interp_0.5-1.0_wl432000_tw8.0-100.0__1b_mute_SW_presmooth30d_srw'

outfolder = '../../figures/paper/CF_dv_merge/xcorrs'
indir = '/data/wsd01/st_helens_peter/corrs_response_removed_newgaphandling_longtw/xstations_5_0.5-1.0_wl130.0_1b_SW/'
dvindir = '/data/wsd01/st_helens_peter/dv/new_gap_handling/xstations_td_taper_no_gap_interp_0.5-1.0_wl432000_tw8.0-100.0__1b_mute_SW_presmooth30d_srw'

stack_len_d = 60

###
set_mpl_params()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

os.chdir(indir)

os.makedirs(outfolder, exist_ok=True)

# Just plot all
infiles = glob.glob(os.path.join(indir, '*.h5'))


def plot_data(network, station, channel, cdb, taulim):
    cst = cdb.get_data(
                f'{network}', f'{station}', f'{channel}', 'subdivision')
    dv = read_dv(
        os.path.join(dvindir, f'DV-{network}.{station}.{channel}.npz'))

    corrstack = cst.stack(stack_len=stack_len_d*86400)
    # mute the part that has not been used for dv/v estimation
    for tr in corrstack:
        tr.data[np.all((tr.times()>-4, tr.times() < 4), axis=0)] += np.nan
    cstack = corrstack.stack()[0]

    ii = ~np.isnan(dv.corr)
    val = -100*dv.value[ii]
    corr = dv.corr[ii]

    fig = plt.figure(figsize=(9, 7))
    plt.subplots_adjust(wspace=0.07)

    ax1 = fig.add_subplot(121)

    # vmax = max([abs(tr.data).max() for tr in corrstack])
    vmax = .2

    corrstack.plot(
        ax=ax1, timelimits=taulim, cmap='RdGy_r', vmin=-vmax, vmax=vmax)

    ax2 = fig.add_subplot(122, sharey=ax1)

    t = np.array([dt.datetime for dt in dv.stats.corr_start])[ii]

    # amplification factor for stack
    amp = (UTCDateTime(max(t))-UTCDateTime(min(t)))/(3*cstack.data.max())
    offset = UTCDateTime(min(t)).timestamp + (
        UTCDateTime(max(t)) - UTCDateTime(min(t)))/2
    ystack = cstack.data*amp + offset

    ystack = [UTCDateTime(y).datetime for y in ystack]
    ax1.plot(cstack.times(), ystack, 'k', linewidth=1)
    ax2.set_axisbelow(True)

    ax2.grid(True, 'major', 'both')
    map = plt.scatter(
            val, t, c=corr,
            cmap='inferno_r', vmin=0., vmax=1., s=5)
    plt.xlabel(r"$dv/v$ [%]", fontsize=14)

    # Correct format of X-Axis
    locale.setlocale(locale.LC_ALL, "en_GB.utf8")

    ax1.set_xlim(taulim)
    plt.setp(ax2.get_yticklabels(), visible=False)

    ax1.tick_params(
        direction='out', top=False, right=False, axis='both', which='both')
    ax2.tick_params(
        direction='out', top=False, right=False, axis='both', which='both')
    ax1.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")
    ax2.set_xlim((-.5, .5))

    plt.colorbar(
        map, label='Coherence', shrink=.6, orientation='horizontal')
    plt.ylim((min(t), max(t)))
    

    plt.savefig(os.path.join(outfolder, f'{network}.{station}.{channel}.pdf'),
                dpi=300, facecolor='none', bbox_inches='tight')
    plt.close()


def main():
    if 'xstat' in indir:
        nets, stats = zip(*[
            os.path.basename(infile).split('.')[:-1] for infile in infiles])
        chans = ['']*len(nets)
    else:
        nets, stats, chans = zip(*[
            os.path.basename(infile).split('.')[:-1] for infile in infiles])

    taulimval = 50

    for ii, (network, station, channel) in enumerate(zip(nets, stats, chans)):
        if ii % size != rank:
            continue

        if 'auto' in indir:
            taulim = (0, taulimval)
        else:
            taulim = (-taulimval, taulimval)

        if 'xstat' in indir:
            with CDB(os.path.join(
                    indir, f'{network}.{station}.h5'), mode='r') as cdb:
                # find the available labels
                chans = cdb.get_available_channels(
                    'subdivision', network, station)
                for cha in chans:
                    # rewmember that these channel might not have dvs
                    # due to the clockshift
                    if not os.path.isfile(os.path.join(
                            dvindir, f'DV-{network}.{station}.{cha}.npz')):
                        continue
                    plot_data(network, station, cha, cdb, taulim)
        else:
            with CDB(os.path.join(
                    indir, f'{network}.{station}.{channel}.h5'), mode='r') as cdb:
                # find the available labels
                plot_data(network, station, channel, cdb, taulim)
                

main()
