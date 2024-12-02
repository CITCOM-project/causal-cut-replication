#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=24000
#SBATCH --time=8:00:00

echo sbatch g_estimation.sh $@

configurations=$1
line=$2

module load Anaconda3/5.3.0
source activate tci

python g_estimation.py $(sed -n "${line} p" $configurations)

next=$(($line + 100))
if [ $next -lt $(wc -l < $configurations) ]; then
  sbatch g_estimation.sh $configurations $next
fi

echo "__COMPLETED__"
