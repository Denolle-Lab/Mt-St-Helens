'''
Script to retrieve continuous waveform data on the Cascadia server.
The directory structure adheres to the one SeisMIC employs.

:copyright:
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 9th September 2022 03:07:42 pm
Last Modified: Friday, 9th September 2022 04:51:47 pm
'''

import os
import glob
import warnings

from obspy import UTCDateTime

from pnwstore.mseed import WaveformClient

client = WaveformClient()

starts = [
    UTCDateTime(2004, 10, 2), UTCDateTime(1997, 6, 1),
    UTCDateTime(1997, 6, 1), UTCDateTime(1997, 6, 1)]

nets = ['CC', 'UW', 'UW', 'UW']
# stats = #['JRO', 'SHW', 'HSR', 'EDM']

output = '/home/pmakus/mt_st_helens/mseed'


end = UTCDateTime.now()

# for net, stat, start in zip(nets, stats, starts):
net = 'UW'
stat = 'SOS'
start = UTCDateTime(1997,6,1)
this_day = start
while end - this_day > 86400:
    if len(glob.glob(os.path.join(output, f'{this_day.year}/{net}/{stat}/*.D', f'{net}.{stat}.*.*.D.{this_day.year}.{str(this_day.julday).zfill(3)}'))):
        # file exists
        this_day += 86400
        continue
    try:
        s = client.get_waveforms(
            network=net, station=stat, channel="?H?",
            year=this_day.year, doy=this_day.julday)
    except KeyError:
        warnings.warn(f'{this_day} not in db for {net}.{stat}')

    this_day += 86400
    for tr in s:
        dir = f'mseed/{this_day.year}/{tr.stats.network}/{tr.stats.station}/{tr.stats.channel}.D'
        os.makedirs(dir, exist_ok=True)
        tr.write(
            os.path.join(dir, f'{net}.{stat}.{tr.stats.location}.{tr.stats.channel}.D.{this_day.year}.{str(this_day.julday).zfill(3)}'), format='MSEED')