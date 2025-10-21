# Temporal Causal Testing
This repository contains the replication package for our paper entitled "Causality-Driven Test Case Minimisation for Cyber-Physical Systems".

## Setup
1. Clone the repository.
1. We use a python virtualenv.
This is not strictly necessary, but you will need Python>=3.10, and may need to adapt the setup and replication instructions accordingly.
```
virtualenv -p python3.10 venv
```
1. Activate the virtual environment:
```
source venv/bin/activate
```
1. Install the dependencies:
```
pip install .
```

## Data Collection

### SWaT
Our first subject is the Secure Water Treatment (SWaT) plant.
Their data sharing agreement does not allow us to redistribute the log data, but it is available from the [iTrust Labs website](https://itrust.sutd.edu.sg/itrust-labs_datasets).
You will need to download the `SWaT_Dataset_Attack_v0.xlsx` dataset and place it in `case-1-swat/data`.
Having done this, you will need to `cd` into `case-1-swat` and run `python long_format_data.py -t 225 -s 15 -o data/data-225.csv data/SWaT_dataset_attack_v0.xlsx` to convert the data to the correct format.

### OpenAPS
Our second subject systems is a [simulator](https://github.com/CITCOM-project/APSDigitalTwin) for the OpenAPS/oref0 artificial pancreas system, which we redistribute as part of this replication package to make it self-contained.
The data we collected from this system for our experiments is available from FIXME.
However, to rerun our data collection scripts oref0 needs to be installed and on your `PATH`.
To do this, please follow the instructions on the [oref0 GitHub repo](https://github.com/openaps/oref0).
This project was executed with oref0 version 0.7.1.

> [!NOTE]
> Rerunning our data collection scripts requires over a month's worth of HPC time and may not produce the exact same traces as we used, since part of the simulation is stochastic.
> This will potentially lead to different failures and different causal test outcomes, but should not affect the overall conclusions of the work.
> Rather than rerunning the data collection, we strongly recommend you download our dataset from FIXME.

## Running Causal Tests

For each case (`case-1-swat` and `case-2-oref0`), simply run `bash setup_and_run.sh $case`.
This will create a file `configurations.txt` in the case directory and execute `g_estimation.sh` with each of these of these configurations.
The results will be saved to `$case/logs/sample_{$size}/ci_{$size}/attack_{$index}.json`.

> [!NOTE]
> N.B. This will take a few days for each system, maybe up to a week.
> If you are on an HPC system using `slurm`, this will run automatically with 100 jobs in the queue.
> You may need to adjust the runtime in g_estimation.sh.

## Plotting
To plot the graphs in the paper, run `python plotting/data_analysis.py $case/logs` for each `case` in [`case-1-swat`, `case-2-oref0`].
This will produce `$case/figures` and `$case/stats` directories that contain the figures and statistical analyses for the subject system.
Filenames should be self-explanatory.

## Pre-Commit Hooks
The data agreement for [OpenAPS](https://openaps.org/outcomes/data-commons/) require that we do not publish patient IDs.
To stop this happening accidentally we have implemented a pre-commit hook that checks that no file (and no file name) contains any string in "naughty_strings.txt".
Obviously, we cannot commit this file, but it simply lists each patient ID one per line.
You do not need access to the OpenAPS data to replicate our experiments, but we strongly recommend that you recreate "naughty_strings.txt" if you are working with this data.

> [!WARNING]
> We do not check binary files (e.g. .xlsx files) or compressed folders (e.g. zip or .tar.gz files).
Commit these files at your own risk.
