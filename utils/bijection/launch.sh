#!/bin/sh

#PBS -l nodes=8
#PBS -q mic
#PBS -l walltime=24:00:00

cat $PBS_NODEFILE | sort -u > my_nodefile.txt
sed -e 's/$/:2/' -i my_nodefile.txt
cat my_nodefile.txt

#pin the processes to different CPUs on each node globally
mpirun_config= "-machinefile my_nodefile.txt  -genv I_MPI_PIN_PROCESSOR_LIST 0,1"
L=5000
for NP in "1" "2" "4" "8" "16"; do
  for V in 8000 16000 32000 64000 128000; do
    mpirun -np $NP $mpirun_config mpiLightPCC -n $V -l $L -m 8 -d 0
    mpirun -np $NP $mpirun_config mpiLightPCC -n $V -l $L -m 8 -d 1
    mpirun -np $NP $mpirun_config mpiLightPCC -n $V -l $L -m 9 -d 0
    mpirun -np $NP $mpirun_config mpiLightPCC -n $V -l $L -m 9 -d 1
    #LightPCC -n $V -l $L -m 4 -d 0
    #LightPCC -n $V -l $L -m 4 -d 1
    #echo "alglib: V=$V L=5000"
    #alglib_pr 5000 $V
  done
done

#for fixed number of vectors
V=16000
for NP in "1" "2" "4" "8" "16"; do
  for L in 4000 8000 16000 32000;  do
    mpirun -np $NP $mpirun_config mpiLightPCC -n $V -l $L -m 8 -d 0
    mpirun -np $NP $mpirun_config mpiLightPCC -n $V -l $L -m 8 -d 1
    mpirun -np $NP $mpirun_config mpiLightPCC -n $V -l $L -m 9 -d 0
    mpirun -np $NP $mpirun_config mpiLightPCC -n $V -l $L -m 9 -d 1
   #echo "alglib_pr_intel: V=$V L=5000"
   #alglib_pr_intel $VL $V
  done
done

