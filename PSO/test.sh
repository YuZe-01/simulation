#!/bin/bash
#SBATCH --job-name=1cpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=debug
#SBATCH --output=./main.out

# init environment
cd /public/home/ynhang/yuze/TVB_Distribution/tvb_data
export PATH=`pwd`/bin:$PATH
export PYTHONPATH=`pwd`/lib/python3.10:`pwd`/lib/python3.10/site-packages

if [ ${LD_LIBRARY_PATH+1} ]; then
  export LD_LIBRARY_PATH=`pwd`/lib:`pwd`/bin:$LD_LIBRARY_PATH
else
  export LD_LIBRARY_PATH=`pwd`/lib:`pwd`/bin
fi
if [ ${LD_RUN_PATH+1} ]; then
  export LD_RUN_PATH=`pwd`/lib:`pwd`/bin:$LD_RUN_PATH
else
  export LD_RUN_PATH=`pwd`/lib:`pwd`/bin
fi
cd ../bin

# run python
cd /public/home/ynhang/yuze/code/multitask
/public/home/ynhang/yuze/TVB_Distribution/tvb_data/bin/python /public/home/ynhang/yuze/code/multitask/main.py
