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
    'UW-UW', 'CC-CC', 'CC-PB', 'UW-UW', 'UW-UW', 'CC-UW', 'CC-PB', 'CC-UW',
    'PB-UW', 'UW-UW', 'UW-UW'
])
stats = np.array([
    'EDM-HSR', 'STD-VALT', 'STD-B202', 'HSR-SHW', 'SHW-SOS', 'JRO-JUN',
    'JRO-B204', 'JRO-HSR', 'B204-STD', 'JUN-STD', 'HSR-SOS'
])

bad_corrs = [
    'JRO-SEP', 'JRO-SUG', 'NED-SEP', 'SEP-STD', 'SEP-SWF2', 'SEP-SWFL',
    'SEP-VALT', 'STD-SUG', 'SUG-SWF2', 'SUG-VALT', 'SEP-B202', 'SEP-B204',
    'SUG-B202', 'SUG-B204' 'NED-SUG', 'SEP-EDM', 'SEP-HSR', 'SEP-JUN',
    'SEP-SHW', 'SEP-SOS', 'SEP-SUG', 'SEP-YEL', 'SUG-EDM', 'SUG-HSR',
    'SUG-JUN', 'SUG-SHW', 'SUG-SOS', 'SWF2-STD'
]

stack_len = 0  # Let's put 60d



comm = MPI.COMM_WORLD
psize = comm.Get_size()
# if psize != len(nets):
#     print(f'Best to execute this with {len(nets)} cores')
rank = comm.Get_rank()

f = 1.0

for f in 0.25*2**np.arange(3):
    tlim = 50
    path = glob.glob(
        f'/data/wsd01/st_helens_peter/corrs_response_removed_longtw/xstations_*_{f}-{f*2}*')[0]
    infiles_all = glob.glob(os.path.join(path, '*.h5'))
    infiles = []
    for infile in infiles_all:
        _, stat, _ = os.path.basename(infile).split('.')
        if not stat in bad_corrs:
            infiles.append(infile)

    njobs = len(infiles)

    pmap = (np.arange(njobs)*psize)/njobs
    pmap = pmap.astype(np.int32)
    ind = pmap == rank
    # ind = np.arange(njobs, dtype=int)[ind]


    outpath = f'/data/wsd01/st_helens_peter/figures/xcorrs_{stack_len}stack/{f}-{f*2}'
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
        plt.close()
        # write stack back
        if co is not None:
            with CorrelationDataBase(infile, corr_options=co) as cdb:
                cdb.add_correlation(cst, tag=f'stack_{stack_len}')
        if stack_len == 0:
            csts.append(cst[0])
    if stack_len == 0:
        csts = comm.allgather(csts)
        if rank == 0:
            cst = CorrStream()
            [cst.extend(c) for c in csts]
            cst.plot(
                type='heatmap', cmap='seismic', timelimits=(-tlim, tlim),
                sort_by='distance', plot_reference_v=True,
            ref_v=[1, 1.5, 2, 3, 4]
            )
            plt.savefig(os.path.join(
                        outpath, f'section_hm.png'), dpi=300, facecolor='none')
            plt.close()
            cst.plot(
            type='section', cmap='seismic', timelimits=(-tlim, tlim),
            sort_by='distance', scalingfactor=0.12, plot_reference_v=True,
            ref_v=[1, 1.5, 2, 3, 4]
            )
            plt.savefig(os.path.join(
                    outpath, f'section.png'), dpi=300, facecolor='none')
            plt.close()