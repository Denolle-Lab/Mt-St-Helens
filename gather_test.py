'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Tuesday, 23rd January 2024 09:20:58 am
Last Modified: Tuesday, 23rd January 2024 09:29:08 am
'''

from calendar import c
from mpi4py import MPI

import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Test gather command with vectors of different sizes
x = np.arange(rank+1)

x = comm.gather(x, root=0)

if rank == 0:
    x = np.unique(np.concatenate(x))
    print(x)