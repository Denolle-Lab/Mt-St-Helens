import os
# OpenBLAS needs to be set for 512 threads
os.environ['OPENBLAS_NUM_THREADS'] = '32'

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
options['net'] = {
    'network': '*',
    'station': [
        'VALT', 'STD', 'JRO', 'B202', 'B204', 'SWF2', 'EDM', 'HSR', 'SHW']}

os.chdir('/data/wsd01/st_helens_peter')

meth = 'xstations'


comm = MPI.COMM_WORLD
psize = comm.Get_size()
rank = comm.Get_rank()


for ii in range(3):
    if ii != 2:
        continue

    f = (1/(2**ii), 2/(2**ii))

    tw_starts = np.arange(14)*4/f[0]
    tw_len = 4/f[0]

    smoothlen_d = 45

    try:
        corrdir = glob.glob(f'corrs_response_removed_longtw/{meth}_*_{f[0]}-{f[1]}*')[0]
    except IndexError:
        if int(f[0])==f[0]:
            f[0] = int(f[0])
        if int(f[1])==f[1]:
            f[1] = int(f[1])
        corrdir = glob.glob(f'corrs_response_removed_longtw/{meth}_*_{f[0]}-{f[1]}*')[0]

    # One data point represents 10 days, should be good enough with such
    # a long smoothing
    date_inc = 864000
    win_len = 864000
    options['dv']['win_len'] = win_len
    options['dv']['date_inc'] = date_inc

    options['dv']['preprocessing'][0]['args']['wsize'] = int(smoothlen_d/(date_inc/86400))  # Less smoothing for more precise drop check

    options['co']['subdir'] = corrdir

    options['dv']['freq_min'] = f[0]
    options['dv']['freq_max'] = f[1]
    options['dv']['dt_ref'] = {'win_inc' : 0, 'method': 'mean', 'percentile': 50}
    options['dv']['plot_vel_change'] = False

    # for tws in tw_starts:
    #     dvdir = f'dv/resp_removed_multitw/{meth}_{f[0]}-{f[1]}_wl{win_len}_tw{tws}-{tw_len}_1b_mute_SW_presmooth{smoothlen_d}d_srw'
    #     options['dv']['subdir'] = dvdir
    #     options['fig_subdir'] = dvdir + '_fig'
    #     options['dv']['tw_start'] = tws
    #     options['dv']['tw_len'] = tw_len
    #     m = Monitor(deepcopy(options))
    #     m.compute_velocity_change_bulk()
    # recompute for the whole coda
    tws = 0
    tw_len += tw_starts.max() - 5

    dvdir = f'dv/resp_removed_multitw/{meth}_{f[0]}-{f[1]}_wl{win_len}_tw{tws}-{tw_len}_1b_mute_SW_presmooth{smoothlen_d}d_srw'
    options['dv']['subdir'] = dvdir
    options['fig_subdir'] = dvdir + '_fig'
    options['dv']['tw_start'] = tws
    options['dv']['tw_len'] = tw_len
    m = Monitor(deepcopy(options))
    m.compute_velocity_change_bulk()
