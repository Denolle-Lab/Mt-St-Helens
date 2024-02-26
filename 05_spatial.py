import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import glob

from mpi4py import MPI
from obspy import UTCDateTime
from obspy.geodetics import degrees2kilometers, locations2degrees
import numpy as np

from seismic.monitor.spatial import DVGrid
from seismic.monitor.dv import read_dv


proj_dir = '/your/project/directory'

# New, slightly larger coverage
lat = [46.05, 46.36]
lon = [-122.45, -122.03]

# Y-extent
y = degrees2kilometers(lat[1]- lat[0])

# X-Extent
x = degrees2kilometers(locations2degrees(lat[0], lon[0], lat[0], lon[1]))

# Resolution / grid spacing of the model
res = 1  # km

# Time-series
delta = 5*24*3600
start = UTCDateTime(year=2007, julday=1).timestamp
end = UTCDateTime(year=2023, julday=240).timestamp
times = np.arange(start, end, delta)

# inversion parameters
# geo-parameters
vel = 2.5  # km/s, Ulberg et al. (2020)
# According to Gabrielli et al. (2020)
#  Q_s = 2*pi*f*mf_path/v , mf_path = Q_s*v/(2*pi*f)
mf_path = vel/(2*np.pi*0.0014*3)
dt = .05  # s  # for the numerical integration


corrupt = ['EDM', 'FL2', 'HSR', 'JUN', 'SHW', 'STD', 'SUG', 'YEL']

corrupt = [
    f'UW-UW.{c}-{c}.EHZ-EHZ' for c in corrupt] + ['CC.CC.SUG-SUG.EHZ-EHZ']


def generate_dvs(indir):
    dvs = glob.glob(os.path.join(indir, '*.npz'))
    for dv in dvs:
        yield read_dv(dv)


comm = MPI.COMM_WORLD
psize = comm.Get_size()
rank = comm.Get_rank()

std_model = 0.004

corr_len = 2

for n in range(3):
    if n % psize != rank:
        continue
    freq0 = 0.25*2**n

    outdir = os.path.join(
        proj_dir,
        f'spatial_{corr_len}_std{std_model}',
        f'{freq0}-{freq0*2}')
    os.makedirs(outdir, exist_ok=True)

    # long dvs
    infiles = os.path. join(
        proj_dir,
        f'dv/*_{freq0}-{freq0*2}_*/*.npz')

    if not len(glob.glob(infiles)):
        raise FileNotFoundError(f'{infiles} contains no files')

    # separately for clock shift
    infiles2 = os.path.join(
        proj_dir,
        f'dv/dv_separately/xstations_{freq0}-{freq0*2}_*/*.npz')

    if not len(glob.glob(infiles2)):
        raise FileNotFoundError(f'{infiles2} contains no files')

    dvs_all = read_dv(infiles)
    dvs_all += read_dv(infiles2)

    dvs_all = [dv for dv in dvs_all if dv.stats.id not in corrupt]

    # create grid
    dvg = DVGrid(lat[0], lon[0], res, x, y, dt, vel, mf_path)

    grids = np.zeros(
        (len(dvg.yaxis), len(dvg.xaxis), len(times)), dtype=np.float64)
    grid_res = np.zeros_like(grids)

    # Compute
    for ii, utc in enumerate(times):
        print(f'working on {UTCDateTime(utc)}.')
        utc = UTCDateTime(utc)
        # Find available dvs at this time
        ti = []
        removed = 0
        # align new dvs to grid
        dvg.align_dvs_to_grid(
            dvs_all, utc, 5, 0.5, outdir)
        # aligned in a previous step + newly aligned
        dvs = [
            dv for dv in dvs_all if dv.dv_processing.get('aligned', False)
            is not False]
        print(f'removed {removed} corrupt dvs')
        try:
            dvg.compute_dv_grid(
                dvs, utc, res, corr_len, std_model, compute_resolution=True)
        except IndexError as e:
            print(e)
            grids[:, :, ii] += np.nan
            grid_res[:, :, ii] += np.nan
            continue
        except np.linalg.LinAlgError as e:
            print(e)
            grids[:, :, ii] += np.nan
            grid_res[:, :, ii] += np.nan
            continue

        # save raw data for joint plot and L-curve analysis
        grids[:, :, ii] = dvg.vel_change
        grid_res[:, :, ii] = dvg.resolution

    np.savez(
        os.path.join(outdir, f'dvdt_3D.npz'), dv=grids, xaxis=dvg.xaxis,
        yaxis=dvg.yaxis, taxis=times, statx=dvg.statx, staty=dvg.staty,
        resolution=grid_res)
