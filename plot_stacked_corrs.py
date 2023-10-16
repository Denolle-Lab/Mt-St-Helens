import os
import glob

from matplotlib import pyplot as plt
from mpi4py import MPI

from seismic.db.corr_hdf5 import CorrelationDataBase

outfolder = '../../figures/autcorrs_resp_removed_qc'
indir = '/data/wsd01/st_helens_peter/corrs_response_removed/autoComponents_5_1.0-2.0_wl70.0_1b'

stack_len_d = 60

###

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

os.chdir(indir)

os.makedirs(outfolder, exist_ok=True)

# Just plot all
infiles = glob.glob(os.path.join(indir, '*.h5'))
nets, stats, chans = zip(*[os.path.basename(infile).split('.')[:-1] for infile in infiles])

for ii, (network, station, channel) in enumerate(zip(nets, stats, chans)):
    if ii % size != rank:
        continue

    with CorrelationDataBase(os.path.join(indir, f'{network}.{station}.{channel}.h5'), mode='r') as cdb:
        # find the available labels
        cst = cdb.get_data(f'{network}', f'{station}', f'{channel}', 'subdivision')

    corrstack = cst.stack(stack_len=stack_len_d*86400)

    plt.figure(figsize=(8, 10))
    ax0 = plt.subplot(2, 1, 1)

    corrstack.plot(timelimits=[-10, 10], cmap='seismic', vmin=-0.7, vmax=0.7, ax=ax0)
    ax0.set_title(f'Autocorrelation {network}.{station}.{channel}')
    ax1 = plt.subplot(2, 1, 2)
    corrstack.plot(scalingfactor=300, timelimits=[-10, 10], type='section', ax=ax1)
    plt.savefig(os.path.join(outfolder, f'{network}.{station}.{channel}.png'), dpi=300)
    plt.close()