import os
# OpenBLAS needs to be set for 512 threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import glob

from copy import deepcopy

import yaml
from mpi4py import MPI

import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read_events
from obspy.core.event.catalog import Catalog


from seismic.monitor.monitor import Monitor

yaml_f = '/home/pmakus/mt_st_helens/Mt-St-Helens/params.yaml'
with open(yaml_f) as file:
    options = yaml.load(file, Loader=yaml.FullLoader)

# General options
options['net'] = {'network': '*', 'station': '*'}

os.chdir('/data/wsd01/st_helens_peter')

methods = ['autoComponents', 'betweenComponents']
# methods = ['xstations']
methods = ['autoComponents']

# Get events to exclude from correlations
# minmag = 0

comm = MPI.COMM_WORLD
psize = comm.Get_size()
rank = comm.Get_rank()

# try:
#     evts = read_events(f'MSH_events_minmag{minmag}.xml')
# except FileNotFoundError:
#     if rank == 0:
#         c = Client('USGS', timeout=240)

#         lat = [45.95, 46.45]
#         lon = [-122.45, -121.96]


#         starttime = UTCDateTime(year=1998, julday=1)
#         endtime = UTCDateTime(year=2023, julday=10)
#         delta = 86400*365
#         rtimes = np.linspace(starttime.timestamp, endtime.timestamp, 12)
#         evts = Catalog()

#         for ii, rtime in enumerate(rtimes):
#             if ii == len(rtimes)-1:
#                 break
#             start = UTCDateTime(rtime)
#             end = UTCDateTime(rtimes[ii+1])
#             print(f'downloading events from {start} to {end}')
#             evts.extend(c.get_events(
#                 starttime=start, endtime=end, minmagnitude=minmag, maxdepth=15,
#                 minlatitude=lat[0], maxlatitude=lat[1], minlongitude=lon[0],
#                 maxlongitude=lon[1]))
#         evts.write(f'MSH_events_minmag{minmag}.xml', format='QUAKEML')
#     else:
#         evts = None
#     evts = comm.bcast(evts, root=0)

# otimes = np.array([evt.preferred_origin().time for evt in evts])
for meth in methods:
    for ii in range(3):
        if ii != 0:
            continue
        f = [1/(2**ii), 2/(2**ii)]

        tws = np.floor(4/f[0])
        tw_len = 40/f[0]


        # new standard smoothing for new 0.25, 0.5, and 1.0 Hz
        smoothlen_d = 30
        if meth == 'xstations':
            corrdir = glob.glob(f'corrs_response_removed_longtw/{meth}_*_{f[0]}-{f[1]}*')
        else:
            corrdir = glob.glob(f'corrs_response_removed/{meth}_*_{f[0]}-{f[1]}*')
        if len(corrdir) > 1:
            raise ValueError('ambiguous correlation directory')
        try:
            corrdir = corrdir[0]
        except IndexError:
            if int(f[0])==f[0]:
                f[0] = int(f[0])
            if int(f[1])==f[1]:
                f[1] = int(f[1])
            corrdir = glob.glob(f'corrs_response_removed/{meth}_*_{f[0]}-{f[1]}*')[0]

        # One data point represents 10 days, should be good enough with such
        # a long smoothing
        date_inc = 3600
        win_len = 3600
        options['dv']['win_len'] = win_len
        options['dv']['date_inc'] = date_inc

        dvdir = f'dv/resp_removed_longtw_final_qc/{meth}_{f[0]}-{f[1]}_wl{win_len}_tw{tws}-{tw_len}__1b_mute_SW_presmooth{smoothlen_d}d_mrw91d'

        # options['dv']['preprocessing'].append({
        #     'function': 'pop_at_utcs', 'args': {'utcs': otimes}})
        options['dv']['preprocessing'][0]['args']['wsize'] = int(smoothlen_d/(date_inc/86400))  # Less smoothing for more precise drop check
        options['co']['subdir'] = corrdir
        options['dv']['subdir'] = dvdir
        options['fig_subdir'] = dvdir + '_fig'
        options['dv']['tw_start'] = tws
        options['dv']['tw_len'] = tw_len
        options['dv']['freq_min'] = f[0]
        options['dv']['freq_max'] = f[1]
        options['dv']['dt_ref'] = {'win_inc' : 91.5, 'method': 'mean', 'percentile': 50}

        m = Monitor(deepcopy(options))
        m.compute_velocity_change_bulk()
            # m.compute_components_average(method='AutoComponents')
            # m.compute_components_average(method='CrossComponents')
        # m.compute_components_average(method='CrossStations')
