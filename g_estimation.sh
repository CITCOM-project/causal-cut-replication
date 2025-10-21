#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=24000
#SBATCH --time=8:00:00

echo sbatch g_estimation.sh $@

configurations=$1
line=$2

if command -v module &>/dev/null; then
  module load Anaconda3/5.3.0
fi

source activate tci

python g_estimation.py $(sed -n "${line} p" $configurations)

logfile=$(sed -n "${line}p" $configurations | awk '{print $14}')
case="${logfile%%/*}"
logfile="${logfile#*/}"

cd $case
python scripts/log_postprocessor.py $logfile
cd ..

next=$(($line + 100))
if [ $next -lt $(wc -l < $configurations) ] && command -v sbatch &>/dev/null; then
  sbatch g_estimation.sh $configurations $next
fi

echo "__COMPLETED__"
