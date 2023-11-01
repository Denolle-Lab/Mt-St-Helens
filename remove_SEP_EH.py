'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 29th September 2023 11:16:04 am
Last Modified: Wednesday, 1st November 2023 03:07:01 pm
'''
import glob
import os
import fnmatch
from numpy import delete

from obspy import UTCDateTime

from seismic.db.corr_hdf5 import CorrelationDataBase as CorrDB

networks = ['CC']*4 + ['UW']*3
stations = [
    'NED', 'SEP', 'SEP', 'STD',
    'SOS', 'STD', 'YEL'
    ]
channels = [
    'EHZ', 'EHN', 'EHZ', 'BH?',
    'EHZ', 'EHZ', 'EHZ'
    ]
starttimes = [
    UTCDateTime(2011, 9, 1), UTCDateTime(2015, 1, 1), UTCDateTime(2004,10, 1), UTCDateTime(2006, 1, 1),
    UTCDateTime(2007, 8, 1), UTCDateTime(2002, 4, 1), UTCDateTime(2003, 1, 1)
    ]
endtimes = [
    UTCDateTime(2013,10,1), UTCDateTime(2020, 1, 1), UTCDateTime(2017, 3, 1), UTCDateTime(2017, 10, 1),
    UTCDateTime(2014, 1, 1), UTCDateTime(2013, 11, 1), UTCDateTime(2008, 1, 1)
    ]

infiles = glob.glob(f'/data/wsd01/st_helens_peter/corrs_response_removed_newgaphandling_longtw/xstations*/*SEP*.h5')


def main():
    for infile in infiles:
        if fnmatch.fnmatch(infile, 'CC-*.SEP-*.h5'):
            chafilt = 'EH?-*'
        elif fnmatch.fnmatch(infile, '*-CC.*-SEP.h5'):
            chafilt = '*-EH?'
        net, stat = os.path.basename(infile).split('.')[:-1]
        print(f'working on {infile}')
        delete_corr_from_corrdb(infile, net, stat, chafilt, '*')
    

def delete_corr_from_corrdb(infile, net, stat, chafilt, start):
    with CorrDB(infile, mode='r') as cdb:
        co = cdb.get_corr_options()
        chans = cdb.get_available_channels(
            'subdivision', net, stat)
    print(f'available channels: {chans}')
    for cha in chans:
        if not fnmatch.fnmatch(cha, chafilt):
            print(f'{cha} will not be deleted.')
            continue
        print(f'Deleting {cha} from {infile}...')
        with CorrDB(infile, mode='a', corr_options=co) as cdb:
            cdb.remove_data(net, stat, cha, 'subdivision', start)


main()
