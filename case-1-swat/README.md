# Case 1: Secure Water Treatment (SWaT)

This case study focuses on the [secure water treatment plant](https://itrust.sutd.edu.sg/itrust-labs-home/itrust-labs_swat)] from iTrust labs.
The system is a scaled-down, high-fidelity, industry-compliant emulation of a modern water treatment facility capable of water treatment and purification at a rate of 19 litres per minute.
For our experiment, we used the Attack v0 dataset from SWaT.A1 & A2_Dec 2015.
Due to the terms of the iTrust data sharing agreement, we cannot share the dataset.
If you wish to replicate our study, you can [apply for access](https://itrust.sutd.edu.sg/itrust-labs_datasets) to the data.
If you are granted access, download the v0 attack dataset from SWaT.A1 & A2_Dec 2015 and place it in inside the `data` directory.
The experimental scripts should then work.

## Data preprocessing
The SWaT dataset is one long continuous stream.
Our technique relies on having multiple runs of the software under test.
To simulate this, we segment the data and treat each segment as a separate run.
To ensure that the segments are long enough to perform causal estimation for every attack, we make each segment the length of the longest attack (210 steps).
Since interventions only happen every 15 seconds, we set the timeskip to 15.
To perform the segmentation, run the following command.
```
python long_format_data.py -t 210 -s 15 -o data/data-210.csv data/SWaT_dataset_attack_v0.xlsx
```
This will create a file called `data-210.csv` within the `data` directory.

## Running causal tests
To run our causal tests, simply run the following command from the repository root directory.
```
bash case_1_submissions.sh
```
This will create a directory called `logs` within `case-1-swat` where all of the results will be saved.
