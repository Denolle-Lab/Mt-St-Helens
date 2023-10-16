'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 29th September 2023 11:16:04 am
Last Modified: Friday, 29th September 2023 12:03:59 pm
'''
import glob
import os

from obspy import UTCDateTime

from seismic.db.corr_hdf5 import CorrelationDataBase as CorrDB


freq0 = 0.25


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

for freq0 in [0.25, 0.5, 1.0]:

    infolders = glob.glob(f'/data/wsd01/st_helens_peter/corrs_response_removed/autoComponents*{freq0}-{freq0*2}*')

    filename = '{network}-{network}.{station}-{station}.{channel}-{channel}.h5'

    def main():
        for infolder in infolders:
            print(f'going through {infolder}')
            for net, stat, cha, start, end in zip(
                    networks, stations, channels, starttimes, endtimes):
                infile = os.path.join(
                    infolder, filename.format(network=net, station=stat, channel=cha))
                for infile in glob.glob(infile):
                    print(f'deleting data from {infile}')
                    delete_corr_from_corrdb(infile, net, stat, cha, start, end)
        

    def delete_corr_from_corrdb(infile, net, stat, cha, start, end):
        with CorrDB(infile, mode='r') as cdb:
            co = cdb.get_corr_options()
            chans = cdb.get_available_channels(
                'subdivision', f'{net}-{net}', f'{stat}-{stat}')
        start = UTCDateTime(start)
        end = UTCDateTime(end)
        for cha in chans:
            with CorrDB(infile, mode='a', corr_options=co) as cdb:
                while start < end:
                    thisstart = start.format_fissures()[:-12] + '*'
                    start += 86400
                    cdb.remove_data(f'{net}-{net}', f'{stat}-{stat}', cha, 'subdivision', thisstart)

    main()
