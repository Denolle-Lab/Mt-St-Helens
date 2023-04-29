import os
from glob import glob
import shutil

import numpy as np

from seismic.monitor.dv import read_dv


# if a frac of fraction is below corr_thres remove the velocity change Estimate
corr_thres = 0.4
frac = .75

infolder = '/data/wsd01/st_helens_peter/dv/resp_removed_longtw_final'
outfolder = '/data/wsd01/st_helens_peter/dv/resp_removed_longtw_final_QCpass'

# dv files that pass the QC
qc_pass = []
# files to check
infiles = glob(os.path.join(infolder, '*', '*.npz'))

for infile in infiles:
    
    dv = read_dv(infile)
    iipass = dv.corr[dv.avail] > corr_thres
    if len(dv.value[dv.avail][iipass]) < (1-frac)*len(dv.value[dv.avail]):
        # QC failed
        continue
    # otherwise copy the things to a a pass folder
    outdir = os.path.join(outfolder, os.path.basename(os.path.dirname(infile)))
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir+'_fig', exist_ok=True)
    shutil.copy(infile, outdir)
    # Also copy the figure
    figname = os.path.basename(infile)[3:-4].replace('.', '_')+'.png'
    try:
        shutil.copy(
            os.path.join(os.path.dirname(infile)+'_fig', figname), outdir+'_fig')
    except FileNotFoundError as e:
        print(e)
