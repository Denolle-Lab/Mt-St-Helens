'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 15th February 2024 03:36:05 pm
Last Modified: Thursday, 15th February 2024 04:35:57 pm
'''

import os
import glob
import locale

import mpi4py as MPI
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime

from seismic.db.corr_hdf5 import CorrelationDataBase as CDB
from seismic.monitor.dv import read_dv
from seismic.plot.plot_utils import set_mpl_params

outfolder = '../../figures/paper/CF_dv_merge/autocorrs'
indir = '/data/wsd01/st_helens_peter/corrs_response_removed/autoComponents_5_0.5-1.0_wl120.0_1b'
dvindir = '/data/wsd01/st_helens_peter/dv/new_gap_handling/autoComponents_td_taper_no_gap_interp_0.5-1.0_wl432000_tw8.0-100.0__1b_mute_SW_presmooth30d_srw'

params = {
    # 'font.family': 'Avenir Next',
    'xtick.direction': 'out',
    'ytick.direction': 'out'
}


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
nets, stats, chans = zip(*[os.path.basename(infile).split('.')[:-1] for infile in infiles])

for ii, (network, station, channel) in enumerate(zip(nets, stats, chans)):
    if ii % size != rank:
        continue

    with CDB(os.path.join(indir, f'{network}.{station}.{channel}.h5'), mode='r') as cdb:
        # find the available labels
        cst = cdb.get_data(
            f'{network}', f'{station}', f'{channel}', 'subdivision')
    dv = read_dv(
        os.path.join(dvindir, f'DV-{network}.{station}.{channel}.npz'))

    corrstack = cst.stack(stack_len=stack_len_d*86400)
    cstack = corrstack.stack()

    
    ii = ~np.isnan(dv.corr)
    val = -100*dv.value[ii]
    corr = dv.corr[ii]

    fig = plt.figure(figsize=(8, 7))
    plt.subplots_adjust(wspace=0.07)

    ax1 = fig.add_subplot(121)
    corrstack.plot(ax=ax1, timelimits=(-12, 12), cmap='seismic', vmin=-.5, vmax=.5)

    ax2 = fig.add_subplot(122, sharey=ax1)

    t = np.array([dt.datetime for dt in dv.stats.corr_start])[ii]


    ystack = cstack.data*10000000 + (max(dv.stats.corr_start)-(max(dv.stats.corr_start)-min(dv.stats.corr_start))/2).timestamp
    ystack = [UTCDateTime(y).datetime for y in ystack]
    ax1.plot(cstack.times(), ystack, 'k', linewidth=1)
    map = plt.scatter(
            val , t, c=corr,
            cmap='inferno_r', vmin=0., vmax=1., s=5)
    plt.xlabel(r"$dv/v$ [%]", fontsize=14)

    # Correct format of X-Axis
    locale.setlocale(locale.LC_ALL, "en_GB.utf8")

    ax1.set_xlim((-12, 12))
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)

    ax1.tick_params(direction='out', top=False, right=False, axis='both', which='both')
    ax2.tick_params(direction='out', top=False, right=False, axis='both', which='both')
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")


    plt.colorbar(
        map, label='Coherence', shrink=.6, orientation='horizontal')
    plt.ylim((min(t), max(t)))

    plt.savefig(f'{network}.{station}.{channel}.pdf', dpi=300, facecolor='none')