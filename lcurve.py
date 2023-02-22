import os
import glob
from copy import deepcopy

from mpi4py import MPI
from obspy import UTCDateTime
from obspy.geodetics import degrees2kilometers, locations2degrees
import numpy as np
from matplotlib import pyplot as plt

from seismic.monitor.spatial import DVGrid, data_variance
from seismic.monitor.dv import read_dv


indir = glob.glob('/data/wsd01/st_helens_peter/dv/resp_removed_ddt/xstations_0.5-1.0*')[0]


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
delta = (365.25/2)*24*3600
start = UTCDateTime(year=1999, julday=1).timestamp
end = UTCDateTime(year=2022, julday=1).timestamp
times = np.arange(start, end, delta)

# inversion parameters
# geo-parameters
vel = 1  # km/s
# According to Gabrielli et al. (2020) Q_S^-1 = 0.0014, for 3 Hz, mfp about 38 km
#  Q_s = 2*pi*f*mf_path/v , mf_path = Q_s*v/(2*pi*f)
mf_path = vel/(2*np.pi*0.0014*3)
dt = .05 # s  # for the numerical integration


dvs_all = read_dv(os.path.join(indir, '*.npz'))

# create grid
dvgo = DVGrid(lat[0], lon[0], res, x, y)

comm = MPI.COMM_WORLD
psize = comm.Get_size()
rank = comm.Get_rank()

pmap = (np.arange(len(times))*psize)/len(times)
pmap = pmap.astype(np.int32)
ind = pmap == rank
ind = np.arange(len(times), dtype=int)[ind]


# Computations for L-curve criterion, bi-yearly should be more than sufficient
corr_lens = np.hstack((0.5, np.arange(1, 9, 1)))
stds = [16e-3*(3**n) for n in range(5)]

mv = [] # holds model variances
ress = []  # holds residuals


for std_model in stds:
    for corr_len in corr_lens:
        outdir = os.path.join(
            '/data/wsd01/st_helens_peter/spatial/ddv_dt/',
            f'{os.path.basename(indir)}_cl{corr_len}_std{std_model}_masked')
        os.makedirs(outdir, exist_ok=True)

        # Compute
        residuals = []
        model_variances = []
        for utc in times[ind]:
            utc = UTCDateTime(utc)
            ti = np.argmin(abs(np.array(dvs_all[0].stats.starttime) - utc))
            # Find available dvs at this time
            dvs = [dv for dv in dvs_all if dv.avail[ti]]
            dvg = deepcopy(dvgo)
            try:
                inv = dvg.compute_dv_grid(dvs, utc, dt, vel, mf_path, res, corr_len, std_model)
            except IndexError as e:
                print(e)
                continue
            # Mask the inverse model
            mask = np.zeros(inv.shape)
            mask[:3, :] = True
            mask[:, :4] = True
            mask[-4:, :] = True
            mask[:, -7:] = True
            mask[-9:, -8:] = True
            mask[-7:, -9:] = True
            mask[-5:, -10:] = True
            mask[-6:, :5] = True
            mask[:5, :5] = True
            mask[:4, :6] = True
            mask[:4, -9:] = True
            mask[:5, -8:] = True
            invma = np.ma.array(inv, mask=mask)
            
            real = np.array([dv.value[ti] for dv in dvs])
            corr = np.array([dv.corr[ti] for dv in dvs])
            pred = dvg.forward_model(invma, dt, vel, mf_path, dvs=dvs, utc=utc)

            pp = dvs[0].dv_processing
            sigma_d = data_variance(
                corr, pp['freq_max'] - pp['freq_min'],
                (pp['tw_start'], pp['tw_len'] + pp['tw_start']),
                (pp['freq_max'] + pp['freq_min'])/2)
            residuals.append(np.mean((real - pred)**2/sigma_d**2))

            model_variances.append(np.mean((invma-np.mean(invma))**2))
            # save raw data for joint plot and L-curve analysis
            np.savez(os.path.join(outdir, f'{utc}.npz'), dv=dvg.vel_change, xaxis=dvg.xaxis, yaxis=dvg.yaxis, statx=dvg.statx, staty=dvg.staty)
            plt.figure(figsize=(9, 9))
            ax = plt.gca()
            dvg.plot(ax=ax)
            plt.savefig(os.path.join(outdir, f'{utc}.png'), dpi=300, bbox_inches='tight', transparent=True)
            # plt.savefig(os.path.join(outdir, f'{utc}.pdf'), bbox_inches='tight', transparent=True)
            plt.close()
        mv.append(np.mean(np.hstack(comm.allgather(model_variances))))
        ress.append(np.mean(np.hstack(comm.allgather(residuals))))
if rank == 0:
    np.savez(
        os.path.join(outdir, f'Lcurve.npz'), model_variances=np.array(mv),
        residual=np.array(ress),
        corr_len=np.tile(corr_lens, len(stds)),
        std_model=np.hstack([[std]*len(corr_lens) for std in stds]))
