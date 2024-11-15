# Temporal Causal Testing
This repository contains the replication package for our temporal causal testing paper.

## Pre-requisites
We use an [Anaconda](https://www.anaconda.com/download) virtual environment.
This is not strictly necessary, but you will need Python>=3.10, and will need to adapt the setup and replication instructions accordingly.

## OpenAPS Data Collection
One of our subject systems is a [simulator](https://github.com/CITCOM-project/APSDigitalTwin) for the OpenAPS/oref0 artificial pancreas system, which re redistribute as part of this replication package to make it self-contained.
The data we collected from this system for our experiments is available from FIXME.
However, to rerun our data collection scripts oref0 needs to be installed and on your `PATH`.
To do this, please follow the instructions on the [oref0 GitHub repo](https://github.com/openaps/oref0).
This project was executed with oref0 version 0.7.1.

**Please note** that rerunning our data collection scripts will not produce the exact same traces as we used, since part of the simulation is stochastic.
This will potentially lead to different failures and different causal test outcomes, but should not affect the overall conclusions of the work.

## Running Causal Tests
1. Clone the repository.
1. Create a new virtual environment:
```
conda env create -f environment.yaml --name tci
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


## Pre-Commit Hooks
The data agreement for [OpenAPS](https://openaps.org/outcomes/data-commons/) require that we do not publish patient IDs.
To stop this happening accidentally we have implemented a pre-commit hook that checks that no file (and no file name) contains any string in "naughty_strings.txt".

**CAUTION:** We do not check binary files (e.g. .xlsx files) or compressed folders (e.g. zip or .tar.gz files).
Commit these files at your own risk.
