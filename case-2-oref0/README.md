# Case 2: Artificial pancreas system (OpenAPS)

This case study focuses on the [OpenAPS artificial pancreas system](https://openaps.org), specifically the [oref0 control algorithm](https://github.com/openaps/oref0).
The system is designed to help regulate a user's blood glucose level by injecting them with an appropriate amount of insulin.
We collected attack traces by processing data from the [OpenAPS data commons](https://openaps.org/outcomes/data-commons) which contains data from real users of the system.
The terms of the data sharing agreement do not allow us to make this data available, nor can we publicly share the patient ID of the individual whose trace we used to extract attack traces for the system.
If you wish to replicate this part of our study, please [apply for access](https://openaps.org/outcomes/data-commons) to the data and then contact us for further information.
However, our study is primarily carried out using simulated data, which we will make publicly available and link from this README file once we finish running our experiments.
Please contact us if we forget to do this.

## Data collection
Our test data for this system is too large to distribute as part of this repository.
We will share a link to it from this readme once it is available.
To use it, simply download and extract the zip file within `case-2-oref0`.

If you wish to replicate our data collection process, you must first have oref0 installed on your system.
Please follow the instructions on the [oref0 repo](https://github.com/openaps/oref0) on how to do this.
For this study, we used v0.7.1.
You should then be able to run `python scripts/fuzz_data_generator.py -s $SEED -o fuzz_data -r 5000` from within `case-2-oref0`.
For this study, we used `$SEED=0`, `1`, and `2`.
**Note:** You will first need to do `conda activate tci` to activate conda virtual environment.

## Running causal tests
To run our causal tests, simply run the following command from the repository root directory, with the `tci` conda virtual environment activated.
```
bash case_2_submissions.sh
```
This will create a directory called `logs` within `case-2-swat` where all of the results will be saved.
