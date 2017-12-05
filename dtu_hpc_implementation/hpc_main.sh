#!/bin/sh
# embedded options to qsub - start with #PBS
# -- Name of the job ---
#PBS -N classify_calculations
# -- specify queue --
#PBS -q hpc

# -- estimated wall clock time (execution time): hh:mm:ss --
#PBS -l walltime=03:00:00

# -- number of processors/cores/nodes --
#PBS -l nodes=10:ppn=8

# -- mail notification --
#PBS -m abe

# -- name and placement of erorr and output messages --
#PBS -o hpc_outputs/$USER$PBS_JOBNAME.o$PBS_JOBID
#PBS -e hpc_outputs/$USER$PBS_JOBNAME.e$PBS_JOBID

# -- run in the current working (submission) directory --
if test X$PBS_ENVIRONMENT = XPBS_BATCH; then cd $PBS_O_WORKDIR; fi
# here follow the commands you want to execute

# load modules
module load python/2.7.13
# the following mpi version supports fork()
module load mpi/gcc-6.3.0-openmpi-1.10.6-torque4-testing

# activate the virtual environment which includes the necessary python packages
source ./python_env/bin/activate

# run program - use mpiexec instead of mpirun if possible
# the --mca orte_base_help_aggregate 0 makes the mpi fully verbose (for debuggi$
mpiexec --mca orte_base_help_aggregate 0 python ./main.py --do_weight
