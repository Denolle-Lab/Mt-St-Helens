import os
import glob

from matplotlib import pyplot as plt
import numpy as np

from seismic.monitor.dv import read_dv




for ii in range(3):
    # if ii != 0:
    #     continue
    freq = [1/(2**ii), 2/(2**ii)]
    infolder = glob.glob(
        f'/data/wsd01/st_helens_peter/dv/new_gap_handling/xstations_*_{freq[0]}-{freq[1]}_wl*_*_1b_mute_SW_presmooth60d_srw')[0]
    outfolder = infolder + '_fig_nice'
    os.makedirs(outfolder, exist_ok=True)
    for infile in glob.glob(os.path.join(infolder, '*.npz')):
        dv = read_dv(infile)
        outfile = os.path.join(outfolder, f'{dv.stats.id}.png')
        if os.path.isfile(outfile):
            continue
        try:
            fig, ax = dv.plot(style='publication', return_ax=True,
                        dateformat='%b %y')
            fig.savefig(outfile, dpi=300, bbox_inches='tight')
            plt.close()
        except ValueError as e:
            print(e)
