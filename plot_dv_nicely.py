'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 2nd November 2023 11:52:25 am
Last Modified: Thursday, 2nd November 2023 03:09:19 pm
'''

from math import erfc
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
pressure_data = '/data/wsd01/st_helens_peter/aux_data/climate data/larger_area/pressure_data.npz'


def main():
    for infolder in infolders:
        if 'fig' in infolder:
            continue
        outfolder = infolder + '_fig_pgv_Pc'
        os.makedirs(outfolder, exist_ok=True)
        # get the confining pressure data
        t_P, latv, lonv, confining_pressure, _, _, _ = get_confining_pressure()
        for infile in glob.glob(os.path.join(infolder, '*.npz')):
            plot_dv(infile, outfolder, t_P, confining_pressure, latv, lonv)


def plot_dv(infile, outfolder, t_P, Pc, latv, lonv):
    try:
        dv = read_dv(infile)
        outfile = os.path.join(outfolder, f'{dv.stats.id}.png')
        if os.path.isfile(outfile):
            return
        # get the pgv
        otimes, pgvs = compute_pgv_for_dvv(dv)
        otimes = [ot.datetime for ot in otimes]
        t_P = [t.datetime for t in t_P]
        # Get the confining pressure
        cp = extract_confining_pressure(dv, latv, lonv, Pc)
        # make a two tile subplot, lower tile for the confining pressure
        # We plot the pgvs in the upper tile (same as dv)
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        # plot the dv
        dv.plot(style='publication', ax=ax[0], dateformat='%b %y')
        # plot the pgvs
        ax[0].twinx().bar(otimes, pgvs, width=.1, alpha=.5, color='red')
        # put a label on the twin axis
        ax[0].twinx().set_ylabel('PGV [m/s]')
        ax[1].plot(t_P, cp, 'k')
        ax[1].set_ylabel('Confining pressure [Pa]')
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(e)


def extract_confining_pressure(dv: DV, latv, lonv, confining_pressure):
    # Find out whether we have a cross-correlation or a self-correlation
    if dv.stats.stla == dv.stats.evla and dv.stats.stlo == dv.stats.evlo:
        return extract_confing_pressure_for_coords(
            dv.stats.stla, dv.stats.stlo, latv, lonv, confining_pressure)
    # Otherwise average for the two stations involved
    cp0 = extract_confing_pressure_for_coords(
        dv.stats.stla, dv.stats.stlo, latv, lonv, confining_pressure)
    cp1 = extract_confing_pressure_for_coords(
        dv.stats.evla, dv.stats.evlo, latv, lonv, confining_pressure)
    cp = (cp0 + cp1)/2
    return cp


def extract_confing_pressure_for_coords(
        lat, lon, latv, lonv, confining_pressure):
    # get the closest lat and lon
    lati = np.argmin(np.abs(latv-lat))
    loni = np.argmin(np.abs(lonv-lon))
    # get the confining pressure
    cp = confining_pressure[:, lati, loni]
    return cp


def get_confining_pressure():
    try:
        data = np.load(pressure_data)
        return data['t'], data['latv'], data['lonv'], data['confining_pressure'],\
            data['snow_pressure'], data['pore_pressure'], data['depths']
    except FileNotFoundError:
        pass
    snowmelt, snowfall, snow_depth, precip, latv, lonv = retrieve_weather_data()
    t, latv, lonv, confining_pressure, snow_pressure, pore_pressure, depths =\
        compute_confining_pressure(
            snowmelt, snowfall, snow_depth, precip, latv, lonv)
    np.savez(pressure_data, t=t, latv=latv, lonv=lonv,
             confining_pressure=confining_pressure,
             snow_pressure=snow_pressure, pore_pressure=pore_pressure,
             depths=depths)
    return t, latv, lonv, confining_pressure, snow_pressure, pore_pressure, depths


def compute_confining_pressure(
        snowmelt, snowfall, snow_depth, precip, latv, lonv):
    # check that all the shapes are the same
    assert snowmelt.shape == snowfall.shape == snow_depth.shape == precip.shape
    years = np.arange(1993, 2024)
    # time vector
    t = np.array(
        [datetime(int(years[0]), 1, 1) + i*timedelta(days=1) for i in range(snowmelt.shape[0])])
    water_influx = snowmelt + precip  # in m
    load = np.zeros_like(water_influx)
    # does the diff here make any sense?
    load[1:] = np.diff(water_influx * 1000) * 9.81  # in N/m^2
    # Compute pore pressure changes
    c = 1.  # diffusion coefficient in m^2/s
    dt = 86400.
    # depth in m
    # this should be frequency dependent?
    dmin = 1000
    dmax = 8000
    dstep = 500
    depths = np.arange(dmin, dmax+dstep, dstep)
    # depth dependent pore pressure, add a fourth dimension
    # as a reminder: load is time x lat x lon
    pore_pressure = np.zeros((len(depths), *load.shape))
    for i in range(len(load)):
        X = np.arange(i)
        vals = load[i]
        for r in depths:
            # diffusion
            func = r/np.sqrt(4.*c*(i-X)*dt)
            b = np.sum(vals * erfc(func))
            pore_pressure[r, i] = b

    # remember, we are actually looking at changes
    snow_pressure = np.zeros_like(load)
    # snow_depth is giving in m water equivalent
    snow_pressure[1:] = np.diff(snow_depth * 1000) * 9.81
    # again, this is the confining pressure change
    confining_pressure = snow_pressure - np.mean(pore_pressure, axis=0)
    return t, latv, lonv, confining_pressure, snow_pressure, pore_pressure, depths


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
