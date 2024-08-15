#!/bin/bash
#SBATCH --nodes=1
#sBatch -c=4
#SBATCH --mem=4000
#SBATCH --time=04:00:00
#SBATCH --mail-user=jmafoster1@sheffield.ac.uk

module load Anaconda3/2019.07

source activate tci

echo "sbatch hpc-run.sh "$@

attack_file=$1
outfile=$2
attack_index=$3
data_file=$4

python water_g.py -a $attack_file -o $outfile -A -i $attack_index $data_file
