'''
Removes duplicates between long dv time series and the ones separated at the
time of the clock shift of the analogue stations at MSH.

:copyright:
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Wednesday, 1st November 2023 03:50:16 pm
Last Modified: Wednesday, 1st November 2023 04:07:16 pm
'''

import glob
import os

import numpy as np


separate_dir_early = '/data/wsd01/st_helens_peter/dv/dv_separately/xstations_{fmin}-{fmax}_1997-06-01'
separate_dir_late = '/data/wsd01/st_helens_peter/dv/dv_separately/xstations_{fmin}-{fmax}_2013-10-31'
long_dir = '/data/wsd01/st_helens_peter/dv/new_gap_handling/xstations_td_taper_no_gap_interp_{fmin}-{fmax}_wl432000_*__1b_mute_SW_presmooth30d_srw'

freqmins = 1/2**np.arange(3)

for fmin in freqmins:
    fmax = 2*fmin
    sep_early = glob.glob(separate_dir_early.format(fmin=fmin, fmax=fmax))[0]
    sep_late = glob.glob(separate_dir_late.format(fmin=fmin, fmax=fmax))[0]
    long = glob.glob(long_dir.format(fmin=fmin, fmax=fmax))[0]

    early_dvs = glob.glob(f'{sep_early}/*.npz')
    late_dvs = glob.glob(f'{sep_late}/*.npz')
    
    # delete the files that are not in both lists
    set_early = set([os.path.basename(f) for f in early_dvs])
    set_late = set([os.path.basename(f) for f in late_dvs])
    to_delete = set_early.symmetric_difference(set_late)
    for f in to_delete:
        if f in set_early:
            os.remove(f'{sep_early}/{f}')
        elif f in set_late:
            os.remove(f'{sep_late}/{f}')

    # If they are in early and late remove them from long
    for f in set_early.intersection(set_late):
        try:
            os.remove(f'{long}/{f}')
        except FileNotFoundError:
            # then this has already been executed
            continue