"""
This script computes the dv/v time series used in
Makus et al, 2024 (SRL).

Note that this will compute continuous time series of dv/v. For stations with
clock shifts. Two time series need to be computed around the clock shift
uising the script compute_dv_separately.py.
"""
import os
# OpenBLAS needs to be set for 512 threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from copy import deepcopy

import yaml
from mpi4py import MPI

import numpy as np


from seismic.monitor.monitor import Monitor

yaml_f = './params.yaml'
with open(yaml_f) as file:
    options = yaml.load(file, Loader=yaml.FullLoader)

proj_dir = '/your/project/directory/'
os.chdir(proj_dir)

methods = ['autoComponents', 'betweenComponents', 'xstations']

comm = MPI.COMM_WORLD
psize = comm.Get_size()
rank = comm.Get_rank()

for meth in methods:
    for ii in range(3):
        if ii != 0:
            continue
        f = [1/(2**ii), 2/(2**ii)]

        tws = np.floor(4/f[0])
        tw_len = 50/f[0]

        # new standard smoothing for new 0.25, 0.5, and 1.0 Hz
        corrdir = os.path.join(
            proj_dir, f'corrs_response_removed, {meth}_*_{f[0]}-{f[1]}*')

        dvdir = f'dv/{meth}_{f[0]}-{f[1]}_tw{tws}-{tw_len}'

        options['co']['subdir'] = corrdir
        options['dv']['subdir'] = dvdir
        options['fig_subdir'] = dvdir + '_fig'
        options['dv']['tw_start'] = tws
        options['dv']['tw_len'] = tw_len
        options['dv']['freq_min'] = f[0]
        options['dv']['freq_max'] = f[1]

        m = Monitor(deepcopy(options))
        m.compute_velocity_change_bulk()

