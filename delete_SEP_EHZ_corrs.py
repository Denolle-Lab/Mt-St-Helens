'''
:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 19th September 2023 03:09:49 pm
Last Modified: Friday, 20th October 2023 02:05:53 pm
'''
import glob
import os
import fnmatch

from mpi4py import MPI

from seismic.db.corr_hdf5 import CorrelationDataBase

infile = '/data/wsd01/st_helens_peter/corrs_response_removed_newgaphandlng_longtw/*/*SEP*.h5'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

for ii, file in enumerate(glob.glob(infile)):
    if ii % size != rank:
        continue
    network, station = os.path.basename(file).split('.')[:-1]
    print(file)
    with CorrelationDataBase(file, mode='r') as corrdb:
        co = corrdb.get_corr_options()
        chans = corrdb.get_available_channels('subdivision', network, station)
    if station[:3] == 'SEP':
        remove_chans = fnmatch.filter(chans, 'EHZ-???')
    else:
        remove_chans = fnmatch.filter(chans, '???-EHZ')
    if len(remove_chans) == 0:
        continue
    print(remove_chans)
    with CorrelationDataBase(file, mode='w', corr_options=co) as corrdb:
        for chan in remove_chans:
            corrdb.remove_data(
                network, station, chan, tag='subdivision', corr_start='*')