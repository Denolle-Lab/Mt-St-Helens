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

from seismic.db.corr_hdf5 import CorrelationDataBase
from seismic.correlate.stream import CorrStream

nets = np.array([
    'UW-UW', 'CC-CC', 'CC-PB', 'UW-UW', 'UW-UW'
])
stats = np.array([
    'EDM-HSR', 'STD-VALT', 'STD-B202', 'HSR-SHW', 'SHW-SOS'
])

stack_len = 0  # Let's put 60d

comm = MPI.COMM_WORLD
psize = comm.Get_size()
if psize != 5:
    print('Best to execute this with 5 cores')
rank = comm.Get_rank()

pmap = (np.arange(len(nets))*psize)/len(nets)
pmap = pmap.astype(np.int32)
ind = pmap == rank
ind = np.arange(len(nets), dtype=int)[ind]

for n in range(3):
    f = 1.0/2**n
    tlim = 15/f
    path = glob.glob(
        f'/data/wsd01/st_helens_peter/corrs_response_removed_longtw/xstations_*_{f}-{f*2}*')[0]
    outpath = f'/data/wsd01/st_helens_peter/figures/xcorrs_{stack_len}stack/{f}-{f*2}'
    os.makedirs(outpath, exist_ok=True)
    if stack_len == 0:
        csts = CorrStream()
    for net, stat in zip(nets[ind], stats[ind]):
        outfile = os.path.join(
                outpath, f'{net}.{stat}.Z-Z.png')
        infile = os.path.join(path, f'{net}.{stat}.h5')
        
        with CorrelationDataBase(infile, mode='r') as cdb:
            try:
                cst = cdb.get_data(
                    net, stat, '*', f'stack_{stack_len}')
                co = None
            except KeyError:
                co = cdb.get_corr_options()
                cst = cdb.get_data(net, stat, '??Z-??Z', 'subdivision')
                cst = cst.stack(stack_len=stack_len, regard_location=False)
        if stack_len == 0:
            ax = cst[0].plot(tlim=(-tlim, tlim))
        else:
            ax = cst.plot(type='heatmap', cmap='seismic', timelimits=(-tlim, tlim))
        ax.set_title(f'{net}.{stat}.Z-Z\nStation Distance {cst[0].stats.dist} km')

        plt.savefig(outfile, dpi=300, facecolor='none')
        # write stack back
        if co is not None:
            with CorrelationDataBase(path, corr_options=co) as cdb:
                cdb.add_correlation(cst, tag=f'stack_{stack_len}')
        if stack_len == 0:
            csts.append(cst[0])
    csts = comm.allgather(csts)
    if rank == 0:
        cst = CorrStream()
        [cst.extend(c) for c in csts]
        cst.plot(
            type='section', cmap='seismic', timelimits=(-tlim, tlim),
            sort_by='distance'
        )
        plt.savefig(os.path.join(
                    outpath, f'section.png'), dpi=300, facecolor='none')