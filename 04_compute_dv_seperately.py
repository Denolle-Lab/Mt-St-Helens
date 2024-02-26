'''
The goal of this script is to compute the velocity change between any
of the analogue stations once before and once after the clock shift occured

all other long dvs are computed in comppute_dv.py

:copyright:
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 26th February 2024 10:26:44 am
Last Modified: Monday, 26th February 2024 10:35:56 am
'''

import os
# OpenBLAS needs to be set for 512 threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import glob
from copy import deepcopy
import fnmatch

import yaml
from mpi4py import MPI

import numpy as np
from obspy import UTCDateTime

from seismic.monitor.monitor import Monitor
from seismic.correlate.correlate import compute_network_station_combinations
from seismic.db.corr_hdf5 import CorrelationDataBase as CDB


proj_dir = '/your/proj/dir/'

startdate = UTCDateTime(1998, 1, 1)
split_date = UTCDateTime(2013, 10, 21)
end_date = UTCDateTime(2023, 9, 15)
infolder = os.path.join(proj_dir, 'xstations_?_{freqmin}-{freqmax}_*')


yaml_f = './params.yaml'
with open(yaml_f) as file:
    options = yaml.load(file, Loader=yaml.FullLoader)


# These are the analogue stations. Make sure to compute once after the split
# time and once before the split time
nets = ['UW']*11
stats = [
    'CDF', 'EDM', 'HSR', 'FL2', 'HSR', 'JUN', 'SOS', 'SUG', 'SHW', 'STD',
    'YEL']
combs = compute_network_station_combinations(nets, stats)
unwanted_combs = [
    f'{unet}.{ustat}.h5' for unet, ustat in zip(combs[0], combs[1])]

# calculate all
freqmins = 1/2**np.arange(3)

os.chdir('/data/wsd01/st_helens_peter')

comm = MPI.COMM_WORLD
psize = comm.Get_size()
rank = comm.Get_rank()

for fmin in freqmins:
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
        'start_date': str(startdate),
        'end_date': str(split_date-864000)})
    tw_len = 50/fmin
    tws = 4/fmin

    read_start = deepcopy(options['dv']['start_date'])[:10]
    dvdir = f'dv/dv_separately/xstations_{fmin}-{2*fmin}_{read_start}'

    options['dv']['subdir'] = dvdir
    options['fig_subdir'] = dvdir + '_fig'
    options['dv']['tw_start'] = tws
    options['dv']['tw_len'] = tw_len
    options['dv']['freq_min'] = fmin
    options['dv']['freq_max'] = fmin*2

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
            'start_date': str(split_date + 864000),
            'end_date': str(end_date)})
        read_start = deepcopy(options['dv']['start_date'])[:10]
        dvdir = f'dv/dv_separately/xstations_{fmin}-{2*fmin}_{read_start}'
        options['dv']['subdir'] = dvdir
        options['fig_subdir'] = dvdir + '_fig'

