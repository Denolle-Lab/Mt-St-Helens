'''
Data download for Makus et al., 2024 (SRL).

:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 9th September 2022 11:08:53 am
Last Modified: Monday, 26th February 2024 10:33:04 am
'''

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from seismic.trace_data.waveform import Store_Client

c = Client()

root = '/your/proj_dir/'

stations_CC = [
    'JRO', 'NED', 'REM', 'SEP', 'STD', 'SUG', 'SWF2', 'SWFL', 'VALT']
stations_PB = ['B201', 'B202', 'B203', 'B204']
stations_UW = [
    'CDF', 'EDM', 'FL2', 'HSR', 'JUN', 'REM', 'SEP', 'SHW', 'SOS', 'STD',
    'SUG', 'YEL']
networks = ['CC']*len(stations_CC) + ['PB']*(len(stations_PB)) + ['UW']*len(
    stations_UW)
stations = stations_CC + stations_PB + stations_UW

sc = Store_Client(c, root, read_only=False)
for net, stat in zip(networks, stations):
    start = UTCDateTime(1998, 1, 1)
    end = UTCDateTime(2023, 9, 15)
    sc.download_waveforms_mdl(
        start, end, clients=[c], network=net, station=stat, location='*',
        channel='?H?')
    # note that this might download some unwanted channels. Generally, we
    # want EH?, BH?, and HH?
