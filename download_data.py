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
Last Modified: Thursday, 5th October 2023 03:23:37 pm
'''

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from seismic.trace_data.waveform import Store_Client
import logging

c = Client()

# nets = ['CC', 'UW', 'UW', 'UW', 'UW']
# stats = ['JRO', 'SHW', 'HSR', 'EDM']

# starts = [
#     UTCDateTime(2004, 10, 2), UTCDateTime(1972, 10, 1),
#     UTCDateTime(1985, 8, 12), UTCDateTime(1980, 6, 1)]
# end = UTCDateTime.now()

# logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")

# logger.setLevel(logging.WARNING)
root = '/home/pm/Documents_sync/PhD/StHelens/'
sc = Store_Client(c, root, read_only=False)
# for net, stat, start in zip(nets, stats, starts):
net = 'PB'
stat = 'B203'
start = UTCDateTime(2018, 2, 28)
end = UTCDateTime(2018, 8, 30)
sc.download_waveforms_mdl(
    start, end, clients=[c], network=net, station=stat, location='*',
    channel='EHZ')
