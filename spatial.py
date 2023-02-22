import os
import glob
from copy import deepcopy

from mpi4py import MPI
from obspy import UTCDateTime
from obspy.geodetics import degrees2kilometers, locations2degrees
import numpy as np
from matplotlib import pyplot as plt

from seismic.monitor.spatial import DVGrid
from seismic.monitor.dv import read_dv




# The maps should be like this
lat = [46.09, 46.3]
lon = [-122.34, -122.1]

# Y-extent
y = degrees2kilometers(lat[1]- lat[0])

# X-Extent
x = degrees2kilometers(locations2degrees(lat[0], lon[0], lat[0], lon[1]))

# Resolution
res = 1  # km

# Time-series
delta = 10*24*3600
start = UTCDateTime(year=1998, julday=1).timestamp
end = UTCDateTime(year=2022, julday=365).timestamp
times = np.arange(start, end, delta)

# inversion parameters
# geo-parameters
vel = 1  # km/s
# According to Gabrielli et al. (2020) Q_S^-1 = 0.0014, for 3 Hz, mfp about 38 km
#  Q_s = 2*pi*f*mf_path/v , mf_path = Q_s*v/(2*pi*f)
mf_path = vel/(2*np.pi*0.0014*3)
dt = .05 # s  # for the numerical integration

# needs to be thoroughly tested
corr_len = 1  # km; just a try
std_model = .15  # 3.2e-2



def generate_dvs(indir):
    dvs = glob.glob(os.path.join(indir, '*.npz'))
    for dv in dvs:
        yield read_dv(dv)


comm = MPI.COMM_WORLD
psize = comm.Get_size()
rank = comm.Get_rank()

for n in range(3):
    if n != 2:
        continue
    freq0 = 0.25*2**n

    indir = glob.glob(f'/data/wsd01/st_helens_peter/dv/resp_removed_ddt/xstations_{freq0}-{freq0*2}*')[0]


    dvs_all = read_dv(os.path.join(indir, '*.npz'))

    # create grid
    dvgo = DVGrid(lat[0], lon[0], res, x, y)

    grids = np.zeros(
        (len(dvgo.yaxis), len(dvgo.xaxis), len(times)), dtype=np.float64)



    pmap = (np.arange(len(times))*psize)/len(times)
    pmap = pmap.astype(np.int32)
    ind = pmap == rank
    ind = np.arange(len(times), dtype=int)[ind]


    outdir = os.path.join(
        f'/data/wsd01/st_helens_peter/spatial/ddt_final_cl{corr_len}_std{std_model}',
        f'{os.path.basename(indir)}')
    os.makedirs(outdir, exist_ok=True)

    # Compute
    for ii, utc in zip(ind, times[ind]):
        dvg = deepcopy(dvgo)
        utc = UTCDateTime(utc)
        try:
            dvg.compute_dv_grid(dvs_all, utc, dt, vel, mf_path, res, corr_len, std_model)
        except IndexError as e:
            print(e)
            grids[:, :, ii] += np.nan
            continue
        # save raw data for joint plot and L-curve analysis
        np.savez(os.path.join(outdir, f'{utc}.npz'), dv=dvg.vel_change, xaxis=dvg.xaxis, yaxis=dvg.yaxis, statx=dvg.statx, staty=dvg.staty)
        grids[:, :, ii] = dvg.vel_change
        # plt.figure(figsize=(9, 9))
        # ax = plt.gca()
        # dvg.plot(ax=ax)
        # plt.savefig(os.path.join(outdir, f'{utc}.png'), dpi=300, bbox_inches='tight', transparent=True)
        # plt.savefig(os.path.join(outdir, f'{utc}.pdf'), bbox_inches='tight', transparent=True)
        # plt.close()

    comm.Allreduce(MPI.IN_PLACE, [grids, MPI.DOUBLE], op=MPI.SUM)
    
    if rank == 0:
        np.savez(
            os.path.join(outdir, f'dvdt_3D.npz'), dv=grids, xaxis=dvgo.xaxis,
            yaxis=dvgo.yaxis, taxis=times, statx=dvg.statx, staty=dvg.staty)
    del grids