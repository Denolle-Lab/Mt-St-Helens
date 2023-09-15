from typing import Tuple
import os
import glob

from mpi4py import MPI
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from obspy import UTCDateTime

from seismic.correlate.stream import CorrBulk
from seismic.monitor.monitor import make_time_list
from seismic.db.corr_hdf5 import CorrelationDataBase
from seismic.plot.plot_utils import set_mpl_params


corrdir = '/data/wsd01/st_helens_peter/corrs_response_removed_longtw/xstations_5_1.0-2.0_wl80.0_1b_SW'
lag_times = (5, 50)
win_inc = win_len = 86400*60
outfolder = '/data/wsd01/st_helens_peter/figures/covariance_plots'



def _time_dependent_covariance(data: np.ndarray):
    covariance = np.zeros((data.shape[0], data.shape[0]))

    for i in range(data.shape[0]):
        covariance[i] = np.sum(data[i]*data, axis=1)/np.sum(data[i]**2)
    return covariance


def time_dependent_covariance(
    cb: CorrBulk, lag_times: Tuple[float, float], win_inc: int,
        win_len: int, sides: str = 'both'):
    # trim data
    if sides == 'both':
        cb.trim(-lag_times[1], lag_times[1])
    elif sides == 'left':
        cb.trim(-lag_times[1], 0)
    elif sides == 'right':
        cb.trim(0, lag_times[1])
    else:
        raise ValueError(f'Unknown sides: {sides}')
    # make corr_start and corr_end tables
    starttimes, endtimes = make_time_list(
        cb.stats.corr_start[0], cb.stats.corr_end[-1], win_inc, win_len)
    cb.resample(starttimes, endtimes)
    # get lapse time axis
    t = cb.stats.start_lag + np.arange(cb.stats.npts)*cb.stats.delta
    # which samples to use
    idx = np.where((t >= lag_times[0]) & (t <= lag_times[1]))[0]
    return _time_dependent_covariance(cb.data[:, idx]), starttimes


def time_dependent_covariance_for_station(
    network: str, station: str, corrdir: str,
    lag_times: Tuple[float, float], win_inc: int, win_len: int,
        channel: str = '*'):
    infile = os.path.join(corrdir, f'{network}.{station}.h5')
    if channel == '*':
        with CorrelationDataBase(infile, mode='r') as cdb:
            chans = cdb.get_available_channels('subdivision', network, station)
        cvl = []
        stl = []
        ids = []
        for cha in chans:
            covariance, starttimes, id = time_dependent_covariance_for_station(
                network, station, corrdir, lag_times, win_inc, win_len, cha)
            cvl.extend(covariance)
            stl.extend(starttimes)
            ids.extend(id)
        return cvl, stl, ids
    with CorrelationDataBase(infile, mode='r') as cdb:
        cst = cdb.get_data(network, station, channel, 'subdivision')
    cb = cst.create_corr_bulk()
    covariance, starttimes = time_dependent_covariance(
        cb, lag_times, win_inc, win_len)
    id = cst[0].stats.id
    return [covariance], [starttimes], [id]


def plot_time_dependent_covariance(covariance, starttimes):
    fig, ax = plt.subplots()
    map = ax.imshow(
        covariance, cmap='seismic', vmin=-1, vmax=1, origin='upper',
        extent=mdates.date2num([UTCDateTime(starttimes[0]).datetime, UTCDateTime(starttimes[-1]).datetime, UTCDateTime(starttimes[-1]).datetime, UTCDateTime(starttimes[0]).datetime]))
    # make ten ticks with equally spaced labels between start and end
    ticklabels = [
        UTCDateTime(st).datetime for st in starttimes]
    ax.set_xticks(ticklabels)
    ax.set_yticks(ticklabels)
    # ax.set_xticklabels(ticklabels)
    # ax.set_yticklabels(ticklabels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor')
    plt.colorbar(map)
    ax.yaxis.set_major_locator(mdates.AutoDateLocator())
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%d %h %y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %h %y'))


def main(
    corrdir: str, lag_times: Tuple[float, float], win_inc: int,
        win_len: int, outfolder: str):
    os.makedirs(outfolder, exist_ok=True)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    infiles = glob.glob(os.path.join(corrdir, '*.h5'))
    infiles.sort()
    infiles = np.array_split(infiles, size)[rank]
    set_mpl_params()
    for infile in infiles:
        network, station = os.path.basename(infile).split('.')[0:2]
        covariance, starttimes, ids = time_dependent_covariance_for_station(
            network, station, corrdir, lag_times, win_inc, win_len)
        for cov, st, id in zip(covariance, starttimes, ids):
            plot_time_dependent_covariance(cov, st)
            plt.savefig(os.path.join(outfolder, f'{id}.png'), dpi=300)
            plt.close()
            plot_time_dependent_covariance(np.sign(cov), st)
            plt.savefig(
                os.path.join(outfolder, f'{id}_bin.png'),
                dpi=300, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    main(corrdir, lag_times, win_inc, win_len, outfolder)

