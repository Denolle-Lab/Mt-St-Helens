import os
import glob

from matplotlib import pyplot as plt
import numpy as np

from seismic.monitor.dv import read_dv




# for ii in range(3):
#     # if ii != 0:
#     #     continue
#     freq = [1/(2**ii), 2/(2**ii)]
infolders = glob.glob(
    f'/data/wsd01/st_helens_peter/dv/dv_seperately/xstations_1.0-2.0*') #_td_taper_no_gap_interp_*_wl432000_*_1b_mute_SW_presmooth30d_srw')
for infolder in infolders:
    if 'fig' in infolder:
        continue
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
            plt.close(fig)
        except ValueError as e:
            print(e)
