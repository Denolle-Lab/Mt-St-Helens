import os
from glob import glob

from mpi4py import MPI
import numpy as np
import yaml

from seismic.monitor.monitor import Monitor
from seismic.db.corr_hdf5 import CorrelationDataBase

yaml_f = '../params.yaml'
path2 = '/data/wsd01/st_helens_peter/clusters_response_removed/xstations_{f}-*_1b_SW_cl{n}/{net}.{stat}.h5'

freqs = [0.25*2**n for n in range(3)]
freqs = [0.5]

comm = MPI.COMM_WORLD
psize = comm.Get_size()
rank = comm.Get_rank()
if rank == 0:
    with open(yaml_f) as file:
        options = yaml.load(file, Loader=yaml.FullLoader)
else:
    options = None

options = comm.bcast(options, root=0)

def monitor_cluster(
        n, infile: str, yaml_f: str, f: float, net: str, stat: str):

    options['proj_dir'] = '.'
    dvopt = options['dv']
    dvopt['win_len'] = 86400
    dvopt['date_inc'] = 86400

    if 0.5 >= f >= .25:
        tw_len = 60
    elif f == 1:
        tw_len = 35
    elif f == 2:
        tw_len = 30
    elif f == 4:
        tw_len = 15
    tws = np.floor(7/f)
    smoothlen_d = None

    if f >= 2:
        smoothlen_d = 9
    elif f >= 0.5:
        smoothlen_d = 18
    elif f >= 0.2:
        smoothlen_d = 36
    elif f >= 0.1:
        smoothlen_d = 63
    else:
        smoothlen_d = 90
    
    date_inc = 86400
    win_len = 86400
    options['dv']['win_len'] = win_len
    options['dv']['date_inc'] = date_inc

    options['dv']['dt_ref'] = {'win_inc' : 0, 'method': 'mean', 'percentile': 50}
    options['dv']['preprocessing'][0]['args']['wsize'] = int(smoothlen_d/(date_inc/86400))  # Less smoothing for more precise drop check
    dvopt['freq_min'] = f
    dvopt['freq_max'] = f*2
    dvopt['tw_start'] = tws
    dvopt['tw_len'] = tw_len
    dvdir = os.path.split(os.path.dirname(infile))
    dvdir = os.path.join(
        dvdir[0], 'vel_change',
        f'{dvdir[1]}_{tws}-{tw_len}')
    dvopt['subdir'] = dvdir
    options['fig_subdir'] = dvopt['subdir'] + '_fig'

    dvopt['dt_ref'] = {'win_inc' : 0, 'method': 'mean', 'percentile': 50}
    # This is required because of the multicore hack I'm using
    os.makedirs(dvopt['subdir'], exist_ok=True)
    os.makedirs(options['fig_subdir'], exist_ok=True)

    m = Monitor(options)
    with CorrelationDataBase(infile, mode='r') as cdb:
        channels = cdb.get_available_channels('subdivision', net, stat)
    for channel in channels:
        m.compute_velocity_change(
            infile, 'subdivision', net, stat, channel)



# Execute the script for this
# Very ugly for-loop stacking, but lazy

files = glob('/data/wsd01/st_helens_peter/corrs_response_removed_longtw/xstations*/*.h5')
files = list(set([os.path.basename(f) for f in files]))
networks = [f.split('.')[0] for f in files]
stations = [f.split('.')[1] for f in files]

for net, stat in zip(networks, stations):
    for f in freqs:
        for n in range(4):
            if rank == n and psize >= 4:
                try:
                    inf = glob(
                        path2.format(f=f, net=net, stat=stat, n=n))[0]
                except IndexError:
                    print(f'File {path2.format(f=f, net=net, stat=stat, n=n)} not found.')
                    continue
                monitor_cluster(n, inf, yaml_f, f, net, stat)
            elif rank == 0 and psize < 4:
                print('use at least 4 cores..computing with single core now')
                try:
                    inf = glob(
                        path2.format(f=f, net=net, stat=stat, n=n))[0]
                except IndexError:
                    print('File {inf} not found.')
                    continue
                monitor_cluster(n, inf, yaml_f, f, net, stat)

