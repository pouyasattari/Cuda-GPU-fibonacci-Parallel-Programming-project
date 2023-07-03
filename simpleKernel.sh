#!/bin/bash

if [ $# -ne 1 ]
then
	echo "Usage: $0 <NTHREADS>"
	exit 1
fi

echo "Compile program"
make
echo 

echo "Execute program"
srun -N 1 -n 1 --gpus-per-task=1 -p gpus --reservation=maintenance ./simpleKernel
echo 


echo "Clear temporay file"
make clean
echo 



