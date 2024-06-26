'''
Computes the RMS vaue as QC scheme as in Makus et al. (2024) (SRL).

:copyright:
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 12th September 2023 04:00:52 pm
Last Modified: Monday, 26th February 2024 10:54:33 am
'''
import glob
import os

from matplotlib import dates as mdates
from obspy import read, UTCDateTime, read_inventory
import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI


proj_dir = '/your/project/directory'
infolder = os.path.join(proj_dir, 'mseed/*/*/*/EHZ.D')

# extract available stations
stations = []
for infile in glob.glob(infolder):
    stations.append(os.path.basename(os.path.dirname(infile)))
stations = np.unique(stations)

infolder = '/data/wsd01/st_helens_peter/mseed/*/UW/{station}/EHZ.D/*'
inventory = '/data/wsd01/st_helens_peter/inventory/UW.{station}.xml'

comm = MPI.COMM_WORLD
psize = comm.Get_size()
rank = comm.Get_rank()


def main(plot: bool):
    for ii, station in enumerate(stations):
        outfolder = os.path.join(proj_dir, 'rms')
        os.makedirs(outfolder, exist_ok=True)
        outfile = os.path.join(
            outfolder, 'rms_{station}.npz'.format(station=station))
        if not os.path.exists(outfile):
            rms, starttimes = compute_rms(station)
            if rank == 0:
                np.savez(
                    outfile, rms=rms, starttimes=[st.timestamp for st in starttimes])
        elif plot and rank == 0:
            data = np.load(outfile)
            rms = data['rms']
            starttimes = data['starttimes']
        if plot and rank == 0:
            plot_rms(starttimes, rms, station)
            plt.savefig(
                os.path.join(outfolder, 'rms_{station}.png'.format(station=station)), dpi=300)
            plt.close()


def generate_data(station):
    for infile in glob.glob(infolder.format(station=station)):
        yield read(infile)


def compute_rms(station):
    rms = []
    starttimes = []
    inv = read_inventory(inventory.format(station=station))
    for ii, win in enumerate(generate_data(station)):
        if ii % psize != rank:
            continue
        try:
            win = win.merge()
            win.remove_response(inventory=inv)
            win.detrend()
            win.filter('highpass', freq=0.01)
            rms.append(np.sum(win[0].data**2)**.5)
        except Exception:
            rms.append(np.nan)
        starttimes.append(win[0].stats.starttime)
        if rank == 0:
            print(win[0].stats.starttime.year, win[0].stats.starttime.julday)
    starttimes = comm.reduce(starttimes, op=MPI.SUM, root=0)
    rms = comm.reduce(rms, op=MPI.SUM, root=0)
    if rank == 0:
        rms = [x for _, x in sorted(zip(starttimes, rms))]
        starttimes = sorted(starttimes)
        return rms, starttimes


def plot_rms(starttimes, rms, station):
    starttimes = [UTCDateTime(st).datetime for st in starttimes]
    plt.axhline(np.nanmedian(rms), color='k', label='median')
    plt.axhline(np.nanmean(rms), color='b', label='mean')

    colour = ['g' if x else 'r' for x in  np.array(rms) > .1*np.nanmedian(rms)]
    # change colour of the curve, where it's less than 50% of the median
    plt.scatter(starttimes, rms, color=colour, s=3)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    # mark median and mean of the data
    plt.title(station)
    plt.legend()
    # plt.ylim((0, np.nanmean(rms)))


main(True)
