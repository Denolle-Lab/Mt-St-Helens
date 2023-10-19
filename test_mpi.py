'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 9th February 2023 12:00:05 pm
Last Modified: Thursday, 19th October 2023 09:42:53 am
'''

from mpi4py import MPI
import numpy as np
from obspy import UTCDateTime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

newl = []
counter = [1, 0, 4, 2, 3]
counterl = []
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
for ii, c in enumerate(counter):
    if ii == rank:
        newl.append(alphabet[ii])
        counterl.append(UTCDateTime(c))

# newl = comm.gather(newl, root=0)
newl = comm.reduce(newl, op=MPI.SUM, root=0)
counterl = comm.reduce(counterl, op=MPI.SUM, root=0)
# sort by counter
if rank == 0:
    newl = [x for _, x in sorted(zip(counterl, newl))]
    # newl = newl[::-1]
    counterl = sorted(counterl)
    print(counterl, newl)
# counter = np.array(counter)
# gather all values of counter
# counter = comm.gather(counter, root=0)
# reduce all values of counter and print them afterwards
# counter = comm.reduce(counter, op=MPI.MAX, root=0)
# comm.Allreduce(MPI.IN_PLACE, counter, MPI.MAX)
# print(counter)

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