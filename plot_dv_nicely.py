import os
import glob

from matplotlib import pyplot as plt

from seismic.monitor.dv import read_dv


infolder = '/data/wsd01/st_helens_peter/dv/resp_removed_longtw_final_qc/autoComponents_1.0-2.0_wl3600_tw4.0-40.0__1b_mute_SW_presmooth30d_srw'

outfolder = infolder + '_fig_nice'

os.makedirs(outfolder, exist_ok=True)

for infile in glob.glob(os.path.join(infolder, '*.npz')):
    dv = read_dv(infile)
    fig, ax = dv.plot(style='publication', return_ax=True,
                  dateformat='%b %y')
    fig.savefig(os.path.join(outfolder, f'{dv.stats.id}.png'), dpi=300, bbox_inches='tight')
    plt.close()
