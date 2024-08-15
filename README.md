# Temporal Causal Testing
This repository contains the replication package for our temporal causal testing paper.

## Pre-requisites
We use an [Anaconda](https://www.anaconda.com/download) virtual environment.
This is not strictly necessary, but you will need Python>=3.10, and will need to adapt the setup and replication instructions accordingly.

## Running Causal Tests
1. Clone the repository.
1. Create a new virtual environment:
```
conda create -f environment.yaml --name tci
```
1. Activate the virtual environment:
```
conda activate tci
```
1. For each case (`case-1-swat` and `case-2-oref0`), `cd` into the directory.
1. To run on HPC with SLURM, run `bash hpc-submissions.sh`. To run locally, run `bash local_run.sh`.
Either way, this will create a directory `logs` with one subdirectory for each experimental configuration.
N.B. This will take a few days for `case-1-swat`, maybe up to a week.
To collect a subset of the data, change the range of `for i in {0..30}` to something smaller, e.g. `for i in {0..5}`, or simply kill the process when you run out of time.
1. Process the raw JSON log files into a CSV:
```
python process_results.csv
```
This will create one CSV file within `logs` for each experimental configuration that concatenates the raw JSON log for each attack and aggregates the key information.
