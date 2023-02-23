'''
:copyright:
    Peter Makus
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 22nd February 2023 02:35:08 pm
Last Modified: Wednesday, 22nd February 2023 02:52:17 pm
'''
import glob
import os

from matplotlib import pyplot as plt

from seismic.db.corr_hdf5 import CorrelationDataBase

nets = [
    'UW-UW', 'CC-CC', 'CC-PB', 'UW-UW', 'UW-UW'
]
stats = [
    'EDM-HSR', 'STD-VALT', 'STD-B202', 'HSR-SHW', 'SHW-SOS'
]

for n in range(3):
    f = 1.0/2**n
    path = glob.glob(
        f'/data/wsd01/st_helens_peter/corrs_response_removed_longtw/xstations_*_{f}-{f*2}*')[0]
    outpath = f'/data/wsd01/st_helens_peter/figures/xcorrs/{f}-{f*2}'
    os.makedirs(outpath, exist_ok=True)
    for net, stat in zip(nets, stats):
        outfile = os.path.join(
                outpath, f'{net}.{stat}.Z-Z.png')
        if os.path.isfile(outfile):
            continue
        infile = os.path.join(path, f'{net}.{stat}.h5')
        with CorrelationDataBase(infile, mode='r') as cdb:
            # cst = cdb.get_data('UW-UW', 'HSR-HSR', 'EHZ-EHZ', 'subdivision')
            cst = cdb.get_data(net, stat, '??Z-??Z', 'subdivision')
        cst = cst.stack(stack_len='daily', regard_location=False)
        ax = cst.plot(type='heatmap', cmap='seismic', timelimits=(-15, 15))
        ax.set_title(f'{net}.{stat}.Z-Z\nStation Distance {cst[0].stats.dist} km')
        # plt.savefig(
        #     os.path.join(outpath, f'{net}.{stat}.Z-Z.pdf'), transparent=True)
        plt.savefig(outfile, dpi=300, transparent=True)