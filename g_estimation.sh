#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=24000
#SBATCH --time=24:00:00

echo sbatch g_estimation.sh $@

module load Anaconda3/5.3.0
source activate tci

attacks=$1
dag=$2
safe_ranges=$3
timesteps_per_intervention=$4
confidence=$5
i=$6
outfile=$7
timesteps=$8
baseline_confounders=$9
data=${10}

echo "DATA: $data"

python g_estimation.py \
-a $attacks \
-d $dag \
-s $safe_ranges \
-t $timesteps_per_intervention \
-c $confidence \
-i $i \
-o $outfile \
-b $baseline_confounders \
-T $timesteps \
-S \
$data
