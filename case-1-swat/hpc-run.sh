#!/bin/bash
#SBATCH --nodes=1
#sBatch -c=4
#SBATCH --mem=12000
#SBATCH --time=04:00:00

module load Anaconda3/2019.07

source activate tci

echo "sbatch hpc-run.sh "$@

attack_file=$1
outfile=$2
alpha=$3
attack_index=$4
data_file=$5

python water_g.py -a $attack_file -o $outfile -A -c $alpha -i $attack_index $data_file
