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
Last Modified: Friday, 9th September 2022 03:21:17 pm
'''

import os

from obspy import UTCDateTime

from pnwstore.mseed import WaveformClient

client = WaveformClient()

starts = [
    UTCDateTime(2004, 10, 2), UTCDateTime(1972, 10, 1),
    UTCDateTime(1985, 8, 12), UTCDateTime(1980, 6, 1)]

nets = ['CC', 'UW', 'UW', 'UW', 'UW']
stats = ['JRO', 'SHW', 'HSR', 'EDM']


end = UTCDateTime.now()

for net, stat, start in zip(nets, stats, starts):
    this_day = start
    while end - this_day > 86400:
        s = client.get_waveforms(
            network=net, station=stat, channel="?H?",
            year=this_day.year, doy=this_day.julday)

        this_day += 86400
        for tr in s:
            dir = f'mseed/{this_day.year}/{tr.stats.network}/{tr.stats.station}/{tr.stats.channel}'
            os.makedirs(dir, exist_ok=True)
            tr.write(
                os.path.join(dir, f'{net}.{stat}.{tr.stats.location}.{tr.stats.channel}.{this_day.year}.{this_day.julday}'), format='MSEED')