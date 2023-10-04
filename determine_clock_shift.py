'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 29th September 2023 02:14:28 pm
Last Modified: Wednesday, 4th October 2023 10:26:48 am
'''
import os
import glob
import fnmatch
from copy import deepcopy

from obspy import UTCDateTime
import numpy as np
from mpi4py import MPI

from seismic.db.corr_hdf5 import CorrelationDataBase as CorrDB


# networks and stations that are affected by something that appears
# to be a clock shift
net = 'UW'
stats = [
    'SHW', 'YEL', 'STD', 'SOS', 'JUN', 'HSR', 'FL2', 'EDM'
]
cha = 'EHZ'

skip = ['SUG', 'SEP']

tw = [[tws, twe] for tws, twe in zip(np.arange(9)*5 + 5, np.arange(9)*5 + 20)]

starttime = UTCDateTime(2013, 5, 1)
endtime = UTCDateTime(2014, 2, 1)

# I could raise the precision level by jointly inverted for all frequencies
infolder = '/data/wsd01/st_helens_peter/corrs_response_removed_longtw/xstations_5_1.0-2.0_wl80.0_1b_SW/'

outfolder = '/data/wsd01/st_helens_peter/time_shift_estimates_new'


os.makedirs(outfolder, exist_ok=True)

comm = MPI.COMM_WORLD
psize = comm.Get_size()
rank = comm.Get_rank()


for ii, stat in enumerate(stats):
    if ii % psize != rank:
        continue
    infiles = glob.glob(os.path.join(infolder, f'{net}-*.{stat}-*.h5'))
    infiles += glob.glob(os.path.join(infolder, f'*-{net}.*-{stat}.h5'))
    stats2 = deepcopy(stats)
    stats2.remove(stat)
    # test if the file contains data from two of the affected stations
    for infile in infiles:
        if any([stat2 in infile for stat2 in stats2]) or any(
                [sk in infile for sk in skip]):
            print(f'skipping {infile}')
            continue
        netcode, statcode = os.path.basename(infile).split('.')[:-1]
        with CorrDB(infile, mode='r') as cdb:
            chans = cdb.get_available_channels(
                'subdivision', netcode, statcode)
            av_starts = cdb.get_available_starttimes(
                netcode, statcode, 'subdivision', chans[0]
            )
            av_starts = np.array([UTCDateTime(x) for x in av_starts[chans[0]]])
            if min(abs(av_starts - UTCDateTime(2013, 9, 3))) > 86400:
                print(f'skipping {infile} because of missing data')
                continue
        chans = fnmatch.filter(chans, f'*{cha}*')
        for chacode in chans:
            print(
                f'working on {infile} for {netcode}.{statcode}.{chacode}')
            with CorrDB(infile, mode='r') as cdb:
                try:
                    cst = cdb.get_data(
                        netcode, statcode, chacode, 'subdivision')
                    if cst.count() == 0:
                        raise IndexError
                except (KeyError, IndexError):
                    print(
                        f'For {netcode}.{statcode}.{chacode} no data is '
                        f'available between {starttime} and {endtime}'
                    )
                    continue
            try:
                cb = cst.create_corr_bulk(
                    inplace=True, times=[starttime, endtime], channel=chacode)
            except ValueError as e:
                print(e)
                continue

            # Extract reference trace
            starttimes_new = np.array([starttime + dt for dt in np.arange(
                (endtime-starttime)//3600)*3600]) 
            endtimes_new = starttimes_new + 3600
            cb = cb.resample(starttimes_new, endtimes_new)
            reftr = cb[:90*3600].extract_trace()
            cb.smooth(24*10)
            dt = cb.measure_shift(
                shift_range=1, shift_steps=1001, return_sim_mat=True,
                tw=tw)
            dt.save(os.path.join(outfolder, f'DT-{dt.stats.id}.npz'))
    