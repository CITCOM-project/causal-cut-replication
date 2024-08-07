"""
This module puts timestep data into "long format", i.e. each row represents one
timestep of one run.
"""

import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    prog="long_format_data", description="Converts timestep data to long format."
)
parser.add_argument(
    "-t",
    "--timesteps",
    type=int,
    help="The number of timesteps per run.",
    required=True,
)
parser.add_argument(
    "-s",
    "--timeskip",
    type=int,
    help="The amount of time in between runs starting.",
    required=True,
)
parser.add_argument(
    "-o", "--outfile", type=str, help="Where to save the resulting data.", required=True
)
parser.add_argument("datafiles", nargs="+", help="<Required> Set flag")


actuators = [
    "MV101",
    "P101",
    "P102",
    "MV201",
    "P201",
    "P202",
    "P203",
    "P204",
    "P205",
    "P206",
    "MV302",
    "MV301",
    "P301",
    "P302",
    "P401",
    "P402",
    "P403",
    "P404",
    "UV401",
    "P501",
    "P502",
    # "MV501",
    # "MV502",
    # "MV503",
    "P601",
    "P602",
    "P603",
]


def setup_subject(i, timesteps):
    subject = data.iloc[i : i + timesteps + 1].copy()
    assert len(subject == timesteps)
    subject["id"] = i
    subject["time"] = list(range(timesteps + 1))
    if "Normal/Attack" in subject:
        subject["Attack"] = [x == "Attack" for x in subject["Normal/Attack"]]
        # subject = subject.loc[subject.time % 15 == 0]
        return subject


def build_dataset(data, timesteps, timeskip, outfile):
    # manually check no "object" datatypes
    for k, v in data.dtypes.items():
        print(k.ljust(8), v)

    # Make boolean instead of [0..2] where 0 and 1 are both "off"
    for actuator in actuators:
        data[actuator] = [int(x > 1) for x in data[actuator]]

    individuals = range(0, len(data) - timesteps, timeskip)
    subjects = (setup_subject(i, timesteps) for i in individuals)

    data = pd.concat(subjects)
    if outfile.endswith(".csv"):
        data.to_csv(outfile, index=False)
    elif outfile.endswith(".pqt"):
        data.to_parquet(outfile, index=False)
    else:
        raise ValueError(f"Invalid outfile {outfile}. Must be .csv or .pqt")


# The good datafiles are data/SWaT_Dataset_Attack_v0.csv and data/SWaT_Dataset_Normal_v0.csv
if __name__ == "__main__":
    args = parser.parse_args()

    if args.timeskip is None:
        args.timeskip = args.timesteps
    data = pd.concat([pd.read_csv(f) for f in args.datafiles])
    build_dataset(data, args.timesteps, args.timeskip, args.outfile)
