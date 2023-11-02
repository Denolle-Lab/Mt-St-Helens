'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 2nd November 2023 11:52:25 am
Last Modified: Thursday, 2nd November 2023 02:02:46 pm
'''

import os
import glob
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
import numpy as np
from obspy import Stream, Inventory, UTCDateTime, read, read_inventory,\
    read_events
from obspy.signal.filter import envelope
from obspy.clients.fdsn import Client, header
import pygrib

from seismic.monitor.dv import read_dv, DV
from seismic.utils.miic_utils import resample_or_decimate



infolders = glob.glob(
    f'/data/wsd01/st_helens_peter/dv/dv_seperately/*')
infolders += glob.glob(
    f'/data/wsd01/st_helens_peter/dv/new_gap_handling/*')

evts = '/data/wsd01/st_helens_peter/aux_data/evts_mag2.xml'
mseeds = '/data/wsd01/st_helens_peter/aux_data/mseeds_pgv'
invdir = '/data/wsd01/st_helens_peter/inventory'
pgvfiles = '/data/wsd01/st_helens_peter/aux_data/pgvs/{net}.{sta}.{cha}.npz'
gribfile = '/data/wsd01/st_helens_peter/aux_data/climate data/larger_area/{measure}{year}.grib'


for infolder in infolders:
    if 'fig' in infolder:
        continue
    outfolder = infolder + '_fig_nice'
    os.makedirs(outfolder, exist_ok=True)
    for infile in glob.glob(os.path.join(infolder, '*.npz')):
        plot_dv(infile, outfolder)






def plot_dv(infile, outfolder):
    dv = read_dv(infile)
    outfile = os.path.join(outfolder, f'{dv.stats.id}.png')
    if os.path.isfile(outfile):
        return
    # get the pgv
    otimes, pgvs = compute_pgv_for_dvv(dv)
    # get the weather data
    try:
        fig, ax = dv.plot(style='publication', return_ax=True,
                    dateformat='%b %y')
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close(fig)
    except ValueError as e:
        print(e)



def compute_confining_pressure(
        snowmelt, snowfall, snow_depth, precip, latv, lonv):
    # check that all the shapes are the same
    assert snowmelt.shape == snowfall.shape == snow_depth.shape == precip.shape
    years = np.arange(1993, 2024)
    # time vector
    t = np.array(
        [datetime(int(years[0]), 1, 1) + i*timedelta(days=1) for i in range(snowmelt.shape[0])])
    water_influx = snowmelt + precip  # in m
    load = water_influx * 1000 * 9.81  # in N/m^2




def retrieve_weather_data():
    years = np.arange(1993, 2024)
    snowmelts = []
    snowdepths = []
    snowfalls = []
    precips = []
    for year in years:
        snowmelt, latv, lonv = open_grib(
            f'../climate data/larger_area/snowmelt{year}.grib')
        snowmelts.append(snowmelt)
        snowfall, latv, lonv = open_grib(
            f'../climate data/larger_area/snowfall_{year}.grib')
        snowfalls.append(snowfall)
        snow_depth, latv, lonv = open_grib(
            f'../climate data/larger_area/snow_depth_{year}.grib')
        snowdepths.append(snow_depth)
        precip, latv, lonv = open_grib(
            f'../climate data/larger_area/precip_{year}.grib')
        precips.append(precip)
    snowmelt = np.vstack(snowmelts)
    snowfall = np.vstack(snowfalls)
    snow_depth = np.vstack(snowdepths)
    precip = np.vstack(precips)
    return snowmelt, snowfall, snow_depth, precip, latv, lonv


def compute_pgv_for_dvv(dvv: DV):
    cat = get_events()
    # Find out whether we have a cross-correlation or a self-correlation
    if dvv.stats.network.split('-')[0] + dvv.stats.station.split('-')[0] ==\
            dvv.stats.network.split('-')[1] + dvv.stats.station.split('-')[1]:
        otimes, pgvs = compute_pgv_for_station_and_evts(
            dvv.stats.network.split('-')[0], dvv.stats.station.split('-')[0],
            dvv.stats.channel.split('-'), cat)
    else:
        otimes0, pgvs0 = compute_pgv_for_station_and_evts(
            dvv.stats.network.split('-')[0], dvv.stats.station.split('-')[0],
            dvv.stats.channel.split('-')[0], cat)
        otimes1, pgvs1 = compute_pgv_for_station_and_evts(
            dvv.stats.network.split('-')[1], dvv.stats.station.split('-')[1],
            dvv.stats.channel.split('-')[1], cat)
        # Find the times that are in both lists
        otimes = list(set(otimes0).intersection(otimes1))
        pgvs = []
        for ot in otimes:
            pgvs.append(
                (pgvs0[otimes0.index(ot)] + pgvs1[otimes1.index(ot)])/2)
    return otimes, pgvs


def compute_pgv_for_station_and_evts(net: str, sta: str, cha: str, cat):
    try:
        out = np.load(
            pgvfiles.format(net=net, sta=sta, cha=cha[:-1]))
        otimes = out['otimes']
        otimes = [UTCDateTime(ot) for ot in otimes]
        pgvs = out['pgvs']
        return otimes, pgvs
    except FileNotFoundError:
        pass
    os.makedirs(os.path.dirname(pgvfiles), exist_ok=True)
    otimes = [e.origins[0].time for e in cat]
    pgvs = []
    otimes_out = []
    for ot in otimes:
        st, inv = get_data_around_utc(
            ot, net, sta, cha, mseeddir=mseeds, invdir=invdir)
        if st:
            otimes_out.append(ot)
            pgvs.append(compute_pgv(st, inv))
    np.savez(
        pgvfiles.format(net=net, sta=sta, cha=cha[:-1]),
        otimes=np.array([ot.format_fissures for ot in otimes_out]),
        pgvs=np.array(pgvs))
    return otimes_out, pgvs


def get_events(evtfile=evts):
    try:
        cat = read_events(evtfile)
    except FileNotFoundError:
        cat = download_events()
        cat.write(evts, format='QUAKEML')
    return cat


def download_events():
    client = Client('IRIS')
    cat = client.get_events(
        starttime=UTCDateTime(1997, 6, 1), endtime=UTCDateTime(2023, 9, 1),
        minmagnitude=2, minlatitude=45.2, maxlatitude=47.2,
        minlongitude=-123.6, maxlongitude=-121.0)
    return cat


def compute_pgv(st: Stream, inv: Inventory) -> float:
    # Check whether data was clipped
    env = []
    for tr in st:
        resp = inv.get_response(
            seed_id=tr.id, datetime=tr.stats.starttime)
        if (resp.instrument_sensitivity.value-abs(tr.data).max())/resp.instrument_sensitivity.value < 0.05:
            print(f'Station {tr.id} was clipped')
        env.append(envelope(tr.data))
    try:
        pgv = np.mean(env, axis=0).max()
    except ValueError:
        npts = [len(e) for e in env]
        pgv = env[np.argmax(npts)].max()
    return pgv


def get_data_around_utc(
    utc: UTCDateTime, network: str, station: str, channel: str,
        client: Client | str = 'IRIS', mseeddir='mseeds', invdir='response',
        inventory=None):
    inv = None
    if inventory:
        inventory = inventory.select(network=network, station=station)
        if len(inventory):
            inv = inventory
    os.makedirs(mseeddir, exist_ok=True)
    os.makedirs(invdir, exist_ok=True)
    outfile = os.path.join(
        mseeddir, f'{network}.{station}.{utc.format_fissures()[:-7]}.mseed')
    invfile = os.path.join(invdir, f'{network}.{station}.xml')
    save_inv = False
    if os.path.isfile(outfile):
        return read(outfile), inv or read_inventory(invfile)
    elif os.path.isfile(invfile):
        inv = inv or read_inventory(invfile)
    else:
        inv = inv or None
        save_inv = True
    if isinstance(client, str):
        client = Client(client)
    try:
        st = client.get_waveforms(
            network, station, location='*', channel=f'{channel[:-1]}?',
            starttime=utc, endtime=utc+900)
    except header.FDSNNoDataException:
        print('No data found, returning empty Stream')
        st = Stream()
    if inv is None:
        inv = client.get_stations(
            network=network, station=station, location='*', channel='*',
            level='response')
    if st.count():
        st = resample_or_decimate(st, 10)
        st.remove_response(inventory=inv, output='VEL')
        st.detrend()
        st.taper(max_length=4, max_percentage=5)
        st = st.filter('highpass', freq=0.01)
        st.write(outfile, format='mseed')
    if save_inv:
        inv.write(invfile, format='STATIONXML')
    return st, inv


def open_grib(gribfile: str):
    grbs = pygrib.open(gribfile)
    # This should be a time series on axis 0
    grb = grbs.select()
    data = np.array([g.values for g in grb])
    # Latitude and longitude grid
    lat, lon = grb[0].latlons()
    # make those vectors
    latv = lat[:, 0]
    lonv = lon[0, :]
    if np.all(lonv>180):
        lonv -= 360
    return data, latv, lonv
