import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'


from matplotlib import pyplot as plt
from mpi4py import MPI
import numpy as np
from obspy import UTCDateTime

from seismic.trace_data.waveform import Store_Client
from seismic.plot.plot_spectrum import plot_spct_series


root = '/data/wsd01/st_helens_peter/'
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

networks = ['CC']*3 + ['UW']*7
stations = ['NED', 'SEP', 'STD', 'EDM', 'FL2', 'HSR', 'JUN', 'SHW', 'SOS', 'STD']

for network, station in zip(networks, stations):

    outfile = os.path.join(root, f'figures/spectrograms/{network}.{station}.Z.npz')


    if not os.path.isfile(outfile):
        starttime = UTCDateTime(1998, 1, 1)
        endtime = UTCDateTime(2022, 12, 31)
        sc = Store_Client('IRIS', root, read_only=True)

        f, t, S = sc.compute_spectrogram(network, station, '??Z', starttime, endtime, 86400, freq_max=5)
        if rank == 0:
            np.savez(
                outfile, f=f, t=np.array([tt.format_fissures() for tt in t]), S=S)
    else:
        out = np.load(outfile, allow_pickle=True)
        f = out['f']
        t = out['t']
        t = np.array([UTCDateTime(tt) for tt in t])
        S = out['S']

    if rank == 0:
        fig = plt.figure(figsize=(9, 7))
        plot_spct_series(S, f, t, log_scale=True, flim=(1, 2))
        plt.title(f'{network}.{station}.E')
        plt.savefig(
            os.path.join(root, f'figures/spectrograms/{network}.{station}.Z.1-2.png'),
            dpi=300)