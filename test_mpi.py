'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 9th February 2023 12:00:05 pm
Last Modified: Thursday, 9th February 2023 12:01:41 pm
'''

from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

test = [ii for ii in np.arange(rank, rank+10, 1)]
test = np.hstack(comm.allgather(test))
if rank == 1:
    print(test)