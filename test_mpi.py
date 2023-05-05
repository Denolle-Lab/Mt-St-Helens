'''
:copyright:
   The PyGLImER development team (makus@gfz-potsdam.de).
:license:
   EUROPEAN UNION PUBLIC LICENCE Version 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Thursday, 9th February 2023 12:00:05 pm
Last Modified: Friday, 21st April 2023 09:34:52 am
'''

from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

counter = 0
for ii in range(5):
    counter += 1
# gather all values of counter
# counter = comm.gather(counter, root=0)
# reduce all values of counter and print them afterwards
counter = comm.reduce(counter, op=MPI.SUM, root=0)
print(counter)