import os
import glob

from matplotlib import pyplot as plt
from mpi4py import MPI

from seismic.db.corr_hdf5 import CorrelationDataBase

outfolder = '../../figures/xcorrs'
indir = '/data/wsd01/st_helens_peter/corrs_response_removed_longtw/xstations_5_1.0-2.0_wl80.0_1b_SW'

stack_len_d = 60

###

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

os.chdir(indir)

os.makedirs(outfolder, exist_ok=True)



files = glob.glob(os.path.join(indir, f'*-CC.*-VALT.h5'))
files.extend(glob.glob(os.path.join(indir, f'CC-*.VALT-*.h5')))
for ii, infile in enumerate(files):
    if ii % size != rank:
        continue
    netcode, statcode, _ = os.path.basename(infile).split('.')

    with CorrelationDataBase(infile, mode='r') as cdb:
        # find the available labels
        cst = cdb.get_data(netcode, statcode, '*', 'subdivision')


    corrstack = cst.stack(stack_len=stack_len_d*86400)



    plt.figure(figsize=(8, 10))
    ax0 = plt.subplot(2, 1, 1)

    corrstack.plot(timelimits=[-10, 10], cmap='seismic', vmin=-0.7, vmax=0.7, ax=ax0)
    ax0.set_title(f'Cross Correlation {netcode}.{statcode}')
    ax1 = plt.subplot(2, 1, 2)
    corrstack.plot(scalingfactor=300, timelimits=[-10, 10], type='section', ax=ax1)
    plt.savefig(os.path.join(outfolder, f'{netcode}.{statcode}.{corrstack[0].stats.channel}.png'), dpi=300)
plt.close()