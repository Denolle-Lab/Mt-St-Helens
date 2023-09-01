'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 9th February 2023 12:00:05 pm
Last Modified: Tuesday, 27th June 2023 10:36:48 am
'''

from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

counter = 0
for ii in range(5):
    counter += 1
# counter = np.array(counter)
# gather all values of counter
# counter = comm.gather(counter, root=0)
# reduce all values of counter and print them afterwards
# counter = comm.reduce(counter, op=MPI.MAX, root=0)
# comm.Allreduce(MPI.IN_PLACE, counter, MPI.MAX)
# print(counter)
counter = comm.gather(counter, root=0)
print(counter)


# from mpi4py import MPI

# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

# data = (rank+1)**2
# data = comm.gather(data, root=0)
# if rank == 0:
#     for i in range(size):
#         assert data[i] == (i+1)**2
#     print(data)
# else:
#     assert data is None