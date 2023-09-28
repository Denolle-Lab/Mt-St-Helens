'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 19th September 2023 03:09:49 pm
Last Modified: Tuesday, 19th September 2023 03:20:11 pm
'''
import glob
import os
import fnmatch

from seismic.db.corr_hdf5 import CorrelationDataBase

infile = '/data/wsd01/st_helens_peter/corrs_response_removed_longtw/*/*VALT*.h5'


for file in glob.glob(infile):
    network, station = os.path.basename(file).split('.')[:-1]
    print(file)
    with CorrelationDataBase(file, mode='r') as corrdb:
        co = corrdb.get_corr_options()
        chans = corrdb.get_available_channels('subdivision', network, station)
    remove_chans = fnmatch.filter(chans, '*LH*')
    if len(remove_chans) == 0:
        continue
    print(remove_chans)
    with CorrelationDataBase(file, mode='w', corr_options=co) as corrdb:
        for chan in remove_chans:
            corrdb.remove_data(
                network, station, chan, tag='subdivision', corr_start='*')