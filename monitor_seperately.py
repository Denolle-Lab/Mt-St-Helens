import os
# OpenBLAS needs to be set for 512 threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import glob
from copy import deepcopy
import fnmatch

import yaml
from mpi4py import MPI

import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read_events
from obspy.core.event.catalog import Catalog

from seismic.monitor.monitor import Monitor
from seismic.correlate.correlate import compute_network_station_combinations
from seismic.db.corr_hdf5 import CorrelationDataBase as CDB



# The goal of this script is to compute the velocity change between any
# of the analogue stations once before and once after the clock shift occured
# These are basically all UW stations plus CC.SEP

startdate = UTCDateTime(1997, 6, 1)
split_date = UTCDateTime(2013, 10, 21)
end_date = UTCDateTime(2023, 9, 15)
infolder = '/data/wsd01/st_helens_peter/corrs_response_removed_newgaphandling_longtw/xstations_?_{freqmin}-{freqmax}_*'


yaml_f = '/home/pmakus/mt_st_helens/Mt-St-Helens/params.yaml'
with open(yaml_f) as file:
    options = yaml.load(file, Loader=yaml.FullLoader)


# These are the analogue stations. Make sure to compute once after the split
# time and once before the split time
nets = ['CC'] + ['UW']*8
stats = ['SEP', 'EDM', 'HSR', 'FL2', 'HSR', 'JUN', 'SOS', 'SHW', 'STD']
combs = compute_network_station_combinations(nets, stats)
unwanted_combs = [f'{unet}.{ustat}.h5' for unet, ustat in zip(nets, stats)]

# calcualte all 

freqmins = 1/2**np.arange(3)

os.chdir('/data/wsd01/st_helens_peter')

comm = MPI.COMM_WORLD
psize = comm.Get_size()
rank = comm.Get_rank()

for fmin in freqmins:
    if fmin == 0.25:
        break
    infolder_now = glob.glob(infolder.format(freqmin=fmin, freqmax=2*fmin))[0]

    # find matching files
    infiles = []
    for net, stat in zip(nets, stats):
        infiles += glob.glob(f'{infolder_now}/{net}-*.{stat}-*.h5')
        infiles += glob.glob(f'{infolder_now}/*-{net}.*-{stat}.h5')
    # remove unwanted combinations
    infiles = [f for f in infiles if os.path.basename(f) not in unwanted_combs]


    options['co']['subdir'] = infolder_now


    options['dv'].update({
        'start_date' : str(startdate),
        'end_date' : str(split_date-864000)})
    tw_len = 50/fmin
    tws = 4/fmin

    # new standard smoothing for new 0.25, 0.5, and 1.0 Hz
    smoothlen_d = 60

    # One data point represents 10 days, should be good enough with such
    # a long smoothing
    date_inc = 864000
    win_len = 864000
    options['dv']['win_len'] = win_len
    options['dv']['date_inc'] = date_inc

    read_start = deepcopy(options['dv']['start_date'])[:10]
    dvdir = f'dv/dv_separately/{read_start}'

    options['dv']['preprocessing'][0]['args']['wsize'] = int(smoothlen_d/(date_inc/86400))  # Less smoothing for more precise drop check
    options['dv']['subdir'] = dvdir
    options['fig_subdir'] = dvdir + '_fig'
    options['dv']['tw_start'] = tws
    options['dv']['tw_len'] = tw_len
    options['dv']['freq_min'] = fmin
    options['dv']['freq_max'] = fmin*2
    options['dv']['dt_ref'] = {'win_inc' : 0, 'method': 'mean', 'percentile': 50}


    for ii in range(2):
        m = Monitor(deepcopy(options))
        for ii, inf in enumerate(infiles):
            # mpi
            if ii % psize != rank:
                continue
            stat = os.path.basename(inf).split('.')[1]
            net = os.path.basename(inf).split('.')[0]
            with CDB(inf, mode='r') as cdb:
                chans = cdb.get_available_channels(
                    'subdivision', net, stat)
            chans = fnmatch.filter(chans, '*EH*')
            for cha in chans:
                m.compute_velocity_change(inf, 'subdivision', net, stat, cha)

        options['dv'].update({
            'start_date' : str(split_date+864000),
            'end_date' : str(end_date)})
        read_start = deepcopy(options['dv']['start_date'])[:10]
        dvdir = f'dv/dv_separately/{read_start}'
        options['dv']['subdir'] = dvdir
        options['fig_subdir'] = dvdir + '_fig'

