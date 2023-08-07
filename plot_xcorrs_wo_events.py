'''
:copyright:
    Peter Makus
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 22nd February 2023 02:35:08 pm
Last Modified: Wednesday, 22nd February 2023 03:51:14 pm
'''
import glob
import os

from matplotlib import pyplot as plt
from mpi4py import MPI
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.core.event.catalog import Catalog
from obspy import read_events

from seismic.db.corr_hdf5 import CorrelationDataBase
from seismic.correlate.stream import CorrStream

nets = np.array([
    'UW-UW', 'CC-CC', 'CC-PB', 'UW-UW', 'UW-UW', 'CC-UW', 'CC-PB', 'CC-UW',
    'PB-UW', 'UW-UW', 'UW-UW'
])
stats = np.array([
    'EDM-HSR', 'STD-VALT', 'STD-B202', 'HSR-SHW', 'SHW-SOS', 'JRO-JUN',
    'JRO-B204', 'JRO-HSR', 'B204-STD', 'JUN-STD', 'HSR-SOS'
])


def filter_corrst_by_cat(cst: CorrStream, cat: Catalog) -> CorrStream:
    otimes = np.array([evt.preferred_origin().time for evt in cat])
    cst_filt = CorrStream()
    for ctr in cst:
        if not np.any(np.all(
            [otimes > ctr.stats.corr_start, otimes<ctr.stats.corr_end],
                axis=0)):
            cst_filt.append(ctr)
    print(
        f'Filtered CorrStream; unfilter length {cst.count()}, filtered: '
        f'{cst_filt.count()}.')
    return cst_filt


comm = MPI.COMM_WORLD
psize = comm.Get_size()
# if psize != len(nets):
#     print(f'Best to execute this with {len(nets)} cores')
rank = comm.Get_rank()

stack_len = 0  # 60*86400  # Let's put 60d

f = 0.25
tlim = 15/f
path = glob.glob(
    f'/data/wsd01/st_helens_peter/corrs_response_removed_longtw/xstations_*_{f}-{f*2}*')[0]
infiles = glob.glob(os.path.join(path, '*.h5'))


minmag = 0
try:
    evts = read_events(f'../MSH_events_minmag{minmag}.xml')
except FileNotFoundError:
    if rank == 0:
        c = Client('USGS', timeout=240)

        lat = [45.95, 46.45]
        lon = [-122.45, -121.96]


        starttime = UTCDateTime(year=1998, julday=1)
        endtime = UTCDateTime(year=2023, julday=10)
        delta = 86400*365
        rtimes = np.linspace(starttime.timestamp, endtime.timestamp, 12)
        # rtimes = np.arange(starttime.timestamp, endtime.timestamp, delta)
        evts = Catalog()



        for ii, rtime in enumerate(rtimes):
            if ii == len(rtimes)-1:
                break
            start = UTCDateTime(rtime)
            end = UTCDateTime(rtimes[ii+1])
            print(f'downloading events from {start} to {end}')
            evts.extend(c.get_events(
                starttime=start, endtime=end, minmagnitude=minmag, maxdepth=15,
                minlatitude=lat[0], maxlatitude=lat[1], minlongitude=lon[0],
                maxlongitude=lon[1]))
        evts.write(f'../MSH_events_minmag{minmag}.xml', format='QUAKEML')
    else:
        evts = None
    evts = comm.bcast(evts, root=0)


njobs = len(infiles)

pmap = (np.arange(njobs)*psize)/njobs
pmap = pmap.astype(np.int32)
ind = pmap == rank
# ind = np.arange(njobs, dtype=int)[ind]


outpath = f'/data/wsd01/st_helens_peter/figures/xcorrs_{stack_len}stack_woeq_minmag{minmag}/{f}-{f*2}'
os.makedirs(outpath, exist_ok=True)
if stack_len == 0:
    csts = CorrStream()
# for net, stat in zip(nets[ind], stats[ind]):
for infile in np.array(infiles)[ind]:
    net, stat, _ = os.path.basename(infile).split('.')
    outfile = os.path.join(
            outpath, f'{net}.{stat}.Z-Z.png')
    # infile = os.path.join(path, f'{net}.{stat}.h5')
    
    with CorrelationDataBase(infile, mode='r') as cdb:
        try:
            cst = cdb.get_data(
                net, stat, '*', f'stack_woeq_minmag{minmag}_{stack_len}')
            co = None
        except KeyError:
            co = cdb.get_corr_options()
            cst_all = cdb.get_data(net, stat, '??Z-??Z', 'subdivision')
            cst = filter_corrst_by_cat(cst_all, evts)
            cst = cst.stack(stack_len=stack_len, regard_location=False)
    if stack_len == 0:
        ax = cst[0].plot(tlim=(-tlim, tlim))
    else:
        ax = cst.plot(type='heatmap', cmap='seismic', sort_by='corr_start', timelimits=(-tlim, tlim))
    ax.set_title(f'{net}.{stat}.Z-Z\nStation Distance {cst[0].stats.dist} km')

    plt.savefig(outfile, dpi=300, facecolor='none')
    # write stack back
    if co is not None:
        with CorrelationDataBase(infile, corr_options=co) as cdb:
            cdb.add_correlation(cst, tag=f'stack_woeq_minmag{minmag}_{stack_len}')
    if stack_len == 0:
        csts.append(cst[0])
if stack_len == 0:
    csts = comm.allgather(csts)
if rank == 0 and stack_len == 0:
    cst = CorrStream()
    [cst.extend(c) for c in csts]
    cst.plot(
        type='section', cmap='seismic', timelimits=(-tlim, tlim),
        sort_by='distance', scalingfactor=0.1, plot_reference_v=True,
        ref_v=[1, 1.5, 2]
    )
    plt.savefig(os.path.join(
                outpath, f'section.png'), dpi=300, facecolor='none')
    cst.plot(
        type='heatmap', cmap='seismic', timelimits=(-tlim, tlim),
        sort_by='distance', plot_reference_v=True,
        ref_v=[1, 1.5, 2]
    )
    plt.savefig(os.path.join(
                outpath, f'section_hm.png'), dpi=300, facecolor='none')