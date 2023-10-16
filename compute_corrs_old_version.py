# Execute this script using (in bash) for multicore
# $mpirun python this_script.py

import os
# OpenBLAS needs to be set for 512 threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import yaml
from obspy.clients.fdsn import Client
from mpi4py import MPI



from seismic.correlate.correlate import Correlator
from seismic.trace_data.waveform import Store_Client

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    client = Client('IRIS')
else:
    client = None
client = comm.bcast(client, root=0)


yaml_f = '/home/pmakus/mt_st_helens/Mt-St-Helens/params_old.yaml'
root = '/data/wsd01/st_helens_peter'

# Client is not needed if read_only
sc = Store_Client(client, root, read_only=True)

# Do the actual computation
c = Correlator(sc, yaml_f)
c.pxcorr()
