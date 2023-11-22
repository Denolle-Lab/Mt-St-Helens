'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 9th February 2023 12:00:05 pm
Last Modified: Wednesday, 22nd November 2023 12:06:00 pm
'''

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


tasks = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
# print these with different cores

for ii, t in enumerate(tasks):
    if ii % size == rank:
        print('Rank: ', rank, ' Task: ', t)