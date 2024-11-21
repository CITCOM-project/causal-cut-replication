#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=1000
#SBATCH --time=00:10:00

module load Anaconda3/2019.07
source activate tci

ATTACK=$1
SEED=$2
CMD=bash

if [ $SEED == 0 ]; then
  python scripts/fuzz_data_generator.py -a $ATTACK -s $SEED -o fuzz_data -r 1 -i $SEED -R
else
  python scripts/fuzz_data_generator.py -a $ATTACK -s $SEED -o fuzz_data -r 1 -i $SEED
fi

if command -v sbatch >&2; then
  CMD=sbatch
fi

if [ $SEED -lt 50000 ]; then
  $CMD scripts/collect_fuzz_data.sh $ATTACK $(($SEED + 1))
fi
