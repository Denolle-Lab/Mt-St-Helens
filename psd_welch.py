'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 12th September 2022 10:52:20 am
Last Modified: Monday, 12th September 2022 12:01:19 pm
'''

from typing import Tuple, List
from datetime import datetime
import locale
import os
import warnings
import logging

import numpy as np
from obspy import UTCDateTime, Stream, read_inventory, Trace, read
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
from scipy.signal import welch
from scipy.interpolate import pchip_interpolate


# File locations
# Folder that only contains mseeds same directory structure as the one
# created in downloads
mseed = '/home/pmakus/mt_st_helens/mseed'
# The station xml
stationresponse = '/home/pmakus/mt_st_helens/station/UW.SHW.xml'
# output directories
output = '/home/pmakus/mt_st_helens/PSD'
# Station Name '{network}.{station}'
station = 'SHW'
network = 'UW'
channel = '??Z'
startdate = UTCDateTime(1980, 1, 1)
enddate = UTCDateTime(2021, 12, 31)


# Directory structure
file = '{date.year}/{network}/{station}/{channel}/{network}.{station}.*.{channel}.{date.year}.{date.julday}'

# Batlow colour scale
cm_data = np.loadtxt("batlow.txt")


def main():
    os.makedirs(output, exist_ok=True)
    # Configure logging
    logfile = os.path.join(output, 'log.txt')
    logger = logging.getLogger('compute_psd')
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s"))
    fh = logging.FileHandler(logfile, mode='w')
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s"))
    logger.addHandler(sh)
    logger.addHandler(fh)
    if os.path.isfile(os.path.join(output, f'{network}.{station}.npz')):
        with np.load(os.path.join(output, f'{network}.{station}.npz')) as A:
                l = []
                for item in A.files:
                    l.append(A[item])
                f, t, S = l
    else:
        starts = np.arange(startdate, enddate, 24*3600)
        f, t, S = spct_series_welch(starts, 4*3600, network, station)
    plot_spct_series(S, f, t, )
    plt.savefig(os.path.join(output, f'{network}.{station}.png'), dpi=300)


def plot_spct_series(
    S: np.ndarray, f: np.ndarray, t: np.ndarray, norm: str = None,
    norm_method: str = None, title: str = None, outfile=None, fmt='pdf',
    dpi=300, flim: Tuple[int, int] = None,
        tlim: Tuple[datetime, datetime] = None):
    """
    Plots a spectral series.

    :param S: A spectral series with dim=2.
    :type S: np.ndarray
    :param f: Vector containing frequencies (Hz)
    :type f: np.ndarray
    :param t: Vector containing times (in s)
    :type t: np.ndarray
    :param norm: Normalise the spectrum either on the time axis with
        norm='t' or on the frequency axis with norm='f', defaults to None.
    :type norm: str, optional
    :param norm_method: Normation method to use.
        Either 'linalg' (i.e., length of vector),
        'mean', or 'median'.
    :param title: Plot title, defaults to None
    :type title: str, optional
    :param outfile: location to save the plot to, defaults to None
    :type outfile: str, optional
    :param fmt: Format to save figure, defaults to 'pdf'
    :type fmt: str, optional
    :param flim: Limit Frequency axis and Normalisation to the values
        in the given window
    :type flim: Tuple[int, int]
    :param tlim: Limit time axis to the values in the given window
    :type tlim: Tuple[datetime, datetime]
    """
    # Show dates in English format
    locale.setlocale(locale.LC_ALL, "en_GB.utf8")
    # Create UTC time series
    utc = []
    for pit in t:
        utc.append(UTCDateTime(pit).datetime)
    del t

    set_mpl_params()

    plt.xscale('log')

    if flim is not None:
        plt.ylim(flim)
        ii = np.argmin(abs(f-flim[0]))
        jj = np.argmin(abs(f-flim[1])) + 1
        f = f[ii:jj]
        S = S[ii:jj, :]
    else:
        plt.xlim(10**-1, f.max())

    if tlim is not None:
        plt.xlim(tlim)
        utc = np.array(utc)
        ii = np.argmin(abs(utc-tlim[0]))
        jj = np.argmin(abs(utc-tlim[1]))
        utc = utc[ii:jj]
        S = S[:, ii:jj]

    # Normalise
    if not norm:
        pass
    elif norm == 'f':
        if norm_method == 'linalg':
            S = np.divide(S, np.linalg.norm(S, axis=1)[:, np.newaxis])
        elif norm_method == 'mean':
            S = np.divide(S, np.mean(S, axis=1)[:, np.newaxis])
        elif norm_method == 'median':
            S = np.divide(S, np.median(S, axis=1)[:, np.newaxis])
        else:
            raise ValueError('Normalisation method %s unkown.' % norm_method)
    elif norm == 't':
        if norm_method == 'linalg':
            S = np.divide(S, np.linalg.norm(S, axis=0))
        elif norm_method == 'mean':
            S = np.divide(S, np.mean(S, axis=0))
        elif norm_method == 'median':
            S = np.divide(S, np.median(S, axis=0))
        else:
            raise ValueError('Normalisation method %s unkown.' % norm_method)
    else:
        raise ValueError('Normalisation %s unkown.' % norm)

    cmap = colors.LinearSegmentedColormap.from_list('batlow', cm_data)
    S /= S.max()
    pcm = plt.pcolormesh(
        f, utc, S, shading='gouraud',
        norm=colors.LogNorm(vmin=1e-10, vmax=1e-5), cmap=cmap
        )
    plt.colorbar(
        pcm, label='energy (normalised)', orientation='horizontal', shrink=.6)
    plt.xlabel(r'$f$ [Hz]')
    # plt.xlabel('(dd/mm)')
    ax = plt.gca()
    ax.yaxis.set_major_locator(mdates.AutoDateLocator())
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%h %y'))
    # plt.xticks(rotation='vertical')
    return ax


def spct_series_welch(
        starts: List[UTCDateTime], window_length: float, net: str, stat: str):
    """
    Computes a spectral time series. Each point in time is computed using the
    welch method. Windows overlap by half the windolength. The input stream can
    contain one or several traces from the same station. Frequency axis is
    logarithmic.
    :param st: Input Stream with data from one station.
    :type st: ~obspy.core.Stream
    :param window_length: window length in seconds for each datapoint in time
    :type window_length: int or float
    :return: Arrays containing a frequency and time series and the spectral
        series.
    :rtype: np.ndarray
    """
    l = []
    logger = logging.getLogger('compute_psd')
    for start in starts:
        # windows will overlap with half the window length
        # Hard-corded nperseg so that the longest period waves that
        # can be resolved are around 300s
        loc = os.path.join(
            mseed, file.format(
                date=start, network=net, station=stat, channel=channel))
        try:
            st = read(loc)
        except (FileNotFoundError, Exception):
            logger.warning(f'File not found {loc}.')
            continue
        tr = preprocess(st[0])
        for wintr in tr.slide(window_length=window_length, step=window_length):
            f, S = welch(wintr.data, fs=tr.stats.sampling_rate)
            # interpolate onto a logarithmic frequency space
            # 256 points of resolution in f direction hardcoded for now
            f2 = np.logspace(-3, np.log10(f.max()), 512)
            S2 = pchip_interpolate(f, S, f2)
            l.append(S2)
    S = np.array(l)

    t = np.linspace(
        starts[0].timestamp, starts[-1].timestamp, S.shape[0]
    )
    np.savez(os.path.join(output, f'{network}.{station}.npz'), f2, t, S.T)
    return f2, t, S.T


def resample_or_decimate(
    data: Trace or Stream, sampling_rate_new: int,
        filter=True) -> Stream or Trace:
    """Decimates the data if the desired new sampling rate allows to do so.
    Else the signal will be interpolated (a lot slower).

    :param data: Stream to be resampled.
    :type data: Stream
    :param sampling_rate_new: The desired new sampling rate
    :type sampling_rate_new: int
    :return: The resampled stream
    :rtype: Stream
    """
    if isinstance(data, Stream):
        sr = data[0].stats.sampling_rate
        srl = [tr.stats.sampling_rate for tr in data]
        if len(set(srl)) != 1:
            # differing sampling rates in stream
            for tr in data:
                try:
                    tr = resample_or_decimate(tr, sampling_rate_new, filter)
                except ValueError:
                    warnings.warn(
                        f'Trace {tr} not downsampled. Sampling rate is lower'
                        + ' than requested sampling rate.')
            return data
    elif isinstance(data, Trace):
        sr = data.stats.sampling_rate
    else:
        raise TypeError('Data has to be an obspy Stream or Trace.')

    srn = sampling_rate_new
    if srn > sr:
        raise ValueError('New sampling rate greater than old. This function \
            is only intended for downsampling.')
    elif srn == sr:
        return data

    # Chosen this filter design as it's exactly the same as
    # obspy.Stream.decimate uses
    # Decimation factor
    factor = float(sr)/float(srn)
    if filter and factor <= 16:
        freq = sr * 0.5 / factor
        data.filter('lowpass_cheby_2', freq=freq, maxorder=12)
    elif filter:
        # Use a different filter
        freq = sr * 0.45 / factor
        data.filter('lowpass_cheby_2', freq=freq, maxorder=12)
    if sr/srn == sr//srn:
        return data.decimate(int(sr//srn), no_filter=True)
    else:
        return data.resample(srn)


def preprocess(tr: Trace):
    """
    Some very basic preprocessing on the string in order to plot the spectral
    series. Does the following steps:
    *1. Remove station response*
    *2. Detrend*
    *3. Decimate if sampling rate>50*
    *4. highpass filter with corner period of 300s.*
    :param st: The input Stream, should only contain Traces from one station.
    :type st: ~obspy.core.Stream
    :return: The output stream and station inventory object
    :rtype: ~obspy.core.Stream and ~obspy.core.Inventory
    """

    # Station response already available?
    inv = read_inventory(stationresponse)

    # Downsample to make computations faster
    tr = resample_or_decimate(tr, 25)
    # Remove station responses
    try:
        tr.attach_response(inv)
        tr.remove_response(inventory=inv)
    except ValueError as e:
        logger = logging.getLogger('compute_psd')
        logger.error(
            f'Problem removing the instrument response Original Error {e}')
    # Detrend
    tr.detrend(type='linear')

    # highpass filter
    tr.filter('bandpass', freqmin=0.01, freqmax=12)

    return tr


def set_mpl_params():
    params = {
        #'font.family': 'Avenir Next',
        'pdf.fonttype': 42,
        'font.weight': 'bold',
        'figure.dpi': 150,
        'axes.labelweight': 'bold',
        'axes.linewidth': 1.5,
        'axes.labelsize': 14,
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 13,
        'xtick.direction': 'in',
        'xtick.top': True,  # draw label on the top
        'xtick.bottom': True,  # draw label on the bottom
        'xtick.minor.visible': True,
        'xtick.major.top': True,  # draw x axis top major ticks
        'xtick.major.bottom': True,  # draw x axis bottom major ticks
        'xtick.minor.top': True,  # draw x axis top minor ticks
        'xtick.minor.bottom': True,  # draw x axis bottom minor ticks
        'ytick.labelsize': 13,
        'ytick.direction': 'in',
        'ytick.left': True,  # draw label on the top
        'ytick.right': True,  # draw label on the bottom
        'ytick.minor.visible': True,
        'ytick.major.left': True,  # draw x axis top major ticks
        'ytick.major.right': True,  # draw x axis bottom major ticks
        'ytick.minor.left': True,  # draw x axis top minor ticks
        'ytick.minor.right': True,  # draw x axis bottom minor ticks
        'legend.fancybox': False,
        'legend.frameon': False,
        'legend.loc': 'upper left',
        'legend.numpoints': 2,
        'legend.fontsize': 'large',
        'legend.framealpha': 1,
        'legend.scatterpoints': 3,
        'legend.edgecolor': 'inherit'
    }
    matplotlib.rcParams.update(params)
    # matplotlib.font_manager._rebuild()


main()
