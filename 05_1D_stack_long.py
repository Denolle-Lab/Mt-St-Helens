'''
Computes the 25 years dv/v time series shown in Figure 3 of Makus
et al. (2024, SRL).

:copyright:
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 26th February 2024 02:09:02 pm
Last Modified: Monday, 26th February 2024 02:17:38 pm
'''

# average all dvs for pore pressure test
import os

import numpy as np
from matplotlib import pyplot as plt
from obspy import UTCDateTime

from seismic.correlate.correlate import compute_network_station_combinations
from seismic.monitor.dv import read_dv
from seismic.monitor.monitor import average_components

proj_dir = '/your/project/directory/'
indir = os.path.join(
    proj_dir, 'dv')

stations = ['EDM', 'FL2', 'HSR', 'JUN', 'SHW', 'SOS', 'STD', 'YEL']
networks = ['UW'] * len(stations)

_, statcombs = compute_network_station_combinations(
    networks, stations, method='betweenStations')

freq = 0.25*2**(np.arange(3))

for f in freq:
    dvs = []
    for comb in statcombs:
        try:
            dvs.append(read_dv(
                os.path.join(
                    indir, f'xstations_{f}-{f*2}/DV-UW-UW.{comb}.EHZ-EHZ.npz')))
        except FileNotFoundError as e:
            print(e)
    dv_av = average_components(dvs, save_scatter=False)
    dv_av.save(os.path.join(indir, f'UW_average_{f}.npz'))
    ax = dv_av.plot(
        dateformat='%Y', style='publication', ylim=(-.6, .6),
        title=None, return_ax=True)
    # smooth the dv_v
    dv_av.smooth_sim_mat(365/5)
    ax.plot([t.datetime for t in dv_av.stats.corr_start], -100*dv_av.value), linewidth=2, 
    # origin time of the Nisqually earthquake
    neq = UTCDateTime('2001-02-28 18:54:32').datetime
    plt.vlines(neq, -.6, .6, colors=['g'], linestyles='dashed')
    plt.axvspan(UTCDateTime(2004, 9, 23).datetime, UTCDateTime(2008, 1, 31).datetime, facecolor='r', alpha=0.25)
    plt.savefig(os.path.join(indir, f'UW_av_{f}.png'), dpi=300, facecolor='None')
    plt.close()

