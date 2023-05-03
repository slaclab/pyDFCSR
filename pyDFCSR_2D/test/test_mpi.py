from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()



# work_size = 127 # arbitrary prime number
work_size = 10
work = np.zeros(work_size)

ave, res = divmod(work_size, mpi_size)
count = [ave + 1 if p < res else ave for p in range(mpi_size)]
displ = [sum(count[:p]) for p in range(mpi_size)]
displ = np.array(displ)


start = displ[rank]
local_size = count[rank]
work_local = np.arange(start,start+local_size,dtype=np.float64)

print("local work: {} in rank {}".format(work_local,rank))

comm.Allgatherv(work_local,[work,count,displ,MPI.DOUBLE])
summe = np.empty(1,dtype=np.float64)


comm.Allreduce(np.sum(work_local),summe,op=MPI.SUM)


print("global work: {} in rank {}".format(work,rank))

print("work {} vs {} in rank {}".format(np.sum(work),summe,rank))