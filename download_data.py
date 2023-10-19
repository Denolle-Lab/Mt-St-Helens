'''
I used this code snippet to retrieve waveform data and response information
from FDSN. To get them internally, use the other script.

:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 9th September 2022 11:08:53 am
Last Modified: Friday, 9th September 2022 04:51:42 pm
'''

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from seismic.trace_data.waveform import Store_Client
import logging

c = Client()



# nets = ['CC']*8 + ['PB']*3 + ['UW']*7
# stats = ['NED', 'SEP', 'STD', 'SUG', 'SWFL', 'SWF2', 'VALT', 'JRO']\
#     + ['B202', 'B203', 'B204']\
#     + ['EDM', 'FL2', 'HSR', 'JUN', 'SHW', 'SOS', 'STD']
nets = ['CC']*6 + ['UW'] + ['PB']
stats = ['HOA', 'LOO', 'MAR', 'NFT', 'REM', 'UNFR', 'CDF', 'B201']

start = UTCDateTime(1998, 1, 1)
end = UTCDateTime.now()

logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")

# logger.setLevel(logging.WARNING)
root = '/data/wsd01/st_helens_peter'
sc = Store_Client(c, root, read_only=False)
for net, stat in zip(nets, stats):
    print(
        f'downloading data for {net}.{stat}'
    )
    if stat == 'B201':
        cha = 'EH?'
    else:
        cha = '?H?'
    sc.download_waveforms_mdl(
        start, end, clients=[c], network=net, station=stat, location='*',
        channel=cha)
