'''
:copyright:
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 14th September 2023 11:45:59 am
Last Modified: Friday, 15th September 2023 12:15:11 pm
'''
import requests
import os
import glob
import fnmatch

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read, Stream

from seismic.db.corr_hdf5 import CorrelationDataBase


network = 'UW'
stations = ['FL2', 'HSR', 'JUN', 'SHW', 'SOS', 'STD', 'SUG', 'YEL', 'EDM']
stations = ['CDF']
rms_file = '/data/wsd01/st_helens_peter/rms/rms_{station}.npz'
mustang_url = 'https://services.iris.edu/mustang/measurements/1/query?metric=max_range&net={network}&sta={station}&cha={channel}&output=xml&timewindow=1998-01-01T00:00:00,2023-09-07T00:00:00&nodata=404'
mseed = '/data/wsd01/st_helens_peter/mseed/{year}/{network}/{station}/{channel}.D/{network}.{station}.{location}.{channel}.D.{year}.{jday}'
outfolder = '/data/wsd01/st_helens_peter/qc/'
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
            for starttime in starttimes_mustang + starttimes_rms:
                moveto = filtermseed.format(
                    year=starttime.year, network=network, station=station,
                    channel='EHZ', location='',
                    jday=str(starttime.julday).zfill(3))
                movefrom = glob.glob(mseed.format(
                    year=starttime.year, network=network, station=station,
                    channel='EHZ', location='*',
                    jday=str(starttime.julday).zfill(3)))
                if os.path.isfile(moveto) or len(movefrom) == 0:
                    continue
                os.makedirs(os.path.dirname(moveto), exist_ok=True)
                os.rename(
                    movefrom[0],
                    moveto)
            remove_from_corrdb(
                network, station, 'EHZ', starttimes_mustang + starttimes_rms)


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


def remove_from_corrdb(network, station, channel, starttimes):
    corrdbs = '/data/wsd01/st_helens_peter/corrs_response*/*/*{network}*.*{station}*.h5'
    corrdbs = glob.glob(corrdbs.format(network=network, station=station))
    for corrdb in corrdbs:
        netcomb = os.path.basename(corrdb).split('.')[0]
        statcomb = os.path.basename(corrdb).split('.')[1]
        if (
                netcomb.split('-')[0] == network 
                and statcomb.split('-')[0] ==station
                ):
            chacomb = channel + '-' + '*'
        else:
            chacomb = '*' + '-' + channel
        with CorrelationDataBase(corrdb, mode='r') as cdb:
            co = cdb.get_corr_options()
            chalist = fnmatch.filter(
                cdb.get_available_channels(
                    'subdivision', netcomb, statcomb),
                chacomb)
        with CorrelationDataBase(corrdb, co, 'a') as cdb:
            for chacomb in chalist:
                for starttime in starttimes:
                    startstr = starttime.format_fissures()[:-12] + '*'
                    cdb.remove_data(
                        netcomb, statcomb, chacomb, 'subdivision', startstr)


main(True, False)

