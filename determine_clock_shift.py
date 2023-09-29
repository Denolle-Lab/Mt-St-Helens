'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 29th September 2023 02:14:28 pm
Last Modified: Friday, 29th September 2023 02:42:51 pm
'''
import os
import glob
import fnmatch

from obspy import UTCDateTime
import numpy as np

from seismic.db.corr_hdf5 import CorrelationDataBase as CorrDB


# networks and stations that are affected by something that appears
# to be a clock shift
net = 'UW'
stats = [
    'SHW', 'YEL', 'STD', 'SOS', 'JUN', 'HSR', 'FL2', 'EDM'
]
cha = 'EHZ'

tw = [[tws, twe] for tws, twe in zip(np.arange(10)*5 + 6, np.arange(10)*5 + 11)]

starttime = UTCDateTime(2012, 1, 1)
endtime = UTCDateTime(2014, 3, 1)

# I could raise the precision level by jointly inverted for all frequencies
infolder = '/data/whd02/st_helens_peter_archive/corrs_response_removed_longtw/xstations_5_1.0-2.0_wl80.0_1b_SW/'

outfolder = '/data/wsd01/st_helens_peter/time_shift_estimates'


os.makedirs(outfolder, exist_ok=True)

for stat in stats:
    infiles = glob.glob(os.path.join(infolder, f'{net}-*.{stat}-*.h5'))
    infiles += glob.glob(os.path.join(infolder, f'*-{net}.*-{stat}.h5'))
    # test if the file contains data from two of the affected stations
    for infile in infiles:
        for stat2 in stats:
            if stat2 == stat:
                continue
            if stat2 in infile:
                print(infile)
                continue
        netcode, statcode = os.path.basename(infile).split('.')[:-1]
        with CorrDB(infile, mode='r') as cdb:
            chans = cdb.get_available_channels(
                'subdivision', netcode, statcode)
        chans = fnmatch.filter(chans, f'*{cha}*')
        for chacode in chans:
            with CorrDB(infile, mode='r') as cdb:
                cst = cdb.get_data(
                    netcode, statcode, chacode, 'subdivision',
                    corr_start=starttime, corr_end=endtime)
            cb = cst.create_corr_bulk(inplace=True)

            dt = cb.measure_shift(tw=tw, shift_range=1, shift_steps=1001, return_sim_mat=True)
            dt.save(os.path.join(outfolder, f'DT-{dt.stats.id}.npz'))
    