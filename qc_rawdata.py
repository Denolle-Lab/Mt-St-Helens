'''
:copyright:
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 14th September 2023 11:45:59 am
Last Modified: Friday, 15th September 2023 11:53:05 am
'''
import requests
import os
import datetime

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read, Stream

network = 'UW'
stations = ['FL2', 'HSR', 'JUN', 'SHW', 'SOS', 'STD', 'SUG', 'YEL', 'EDM']
rms_file = '/data/wsd01/st_helens_peter/rms/rms_{station}.npz'
mustang_url = 'https://services.iris.edu/mustang/measurements/1/query?metric=max_range&net={network}&sta={station}&cha={channel}&output=xml&timewindow=1998-01-01T00:00:00,2023-09-07T00:00:00&nodata=404'
mseed = '/data/wsd01/st_helens_peter/mseed/{year}/{network}/{station}/{channel}.D/{network}.{station}.{location}.{channel}.D.{year}.{jday}'
outfolder = '/data/wsd01/st_helens_peter/qc/'
filterfolder = '/data/wsd01/st_helens_peter/mseed_qcfail'
filtermseed = '/data/wsd01/st_helens_peter/mseed_qcfail/{year}/{network}/{station}/{channel}.D/{network}.{station}.{location}.{channel}.D.{year}.{jday}'

def main(filter, plot):
    os.makedirs(outfolder, exist_ok=True)
    for station in stations:
        starttimes_mustang = [
            UTCDateTime(starttime) for starttime in iris_mustang(network, station)]
        starttimes_rms = [UTCDateTime(starttime) for starttime in rms_value_check(network, station)]
        st = Stream()
        if plot:
            for starttime in starttimes_rms:
                try:
                    st.extend(read(mseed.format(year=starttime.year, network=network, station=station, channel='EHZ', location='*', jday=str(starttime.julday).zfill(3))))
                except Exception as e:
                    print(e)
                    continue
            plot_data(st, 'rms')
            st.clear()
            for starttime in starttimes_mustang:
                try:
                    st.extend(read(mseed.format(year=starttime.year, network=network, station=station, channel='EHZ', location='*', jday=str(starttime.julday).zfill(3))))
                except Exception as e:
                    print(e)
                    continue
            plot_data(st, 'mustang')
            st.clear()
        if filter:
            os.makedirs(filterfolder, exist_ok=True)
            for starttime in starttimes_mustang + starttimes_rms:
                os.rename(
                    mseed.format(year=starttime.year, network=network, station=station, channel='EHZ', location='*', jday=str(starttime.julday).zfill(3)),
                    filtermseed.format(year=starttime.year, network=network, station=station, channel='EHZ', location='*', jday=str(starttime.julday).zfill(3)))


def plot_data(st, kind):
    if not st.count():
        print(f'No corrupt data for filter {kind}')
        return
    for tr in st:
        tr.detrend()
        tr.filter('highpass', freq=0.01)
        plt.plot(tr.data[3600*100:3600*200], alpha=0.5)
    plt.title(f'{st.count()} days discarded.')
    plt.savefig(os.path.join(outfolder, f'{tr.id}_{kind}.png'), dpi=300)
    plt.close()

def iris_mustang(network, station, threshold=500):
    r = requests.get(
        mustang_url.format(network=network, station=station, channel='EHZ'),
        allow_redirects=True)
    df = pd.read_xml(r.content)
    return df[df['value'] < threshold]['start'].values


def rms_value_check(network, station):
    values = np.load(rms_file.format(station=station))
    rms = values['rms']
    starttimes = values['starttimes']
    return starttimes[np.where(rms > 300*np.nanmedian(rms))[0]]

main(False, True)

