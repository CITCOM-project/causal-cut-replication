"""
This module puts timestep data into "long format", i.e. each row represents one
timestep of one run.
"""
import pandas as pd
import argparse

parser = argparse.ArgumentParser(prog="long_format_data", description="Converts timestep data to long format.")
parser.add_argument("timesteps", type=int, help="The number of timesteps per run.")
parser.add_argument("-s", "--timeskip", type=int, help="The amount of time in between runs.")
args = parser.parse_args()

if args.timeskip is None:
    args.timeskip = args.timesteps

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

data = pd.concat(
    [
        # pd.read_csv("data/SWaT_Dataset_Normal_v0.csv"),
        # pd.read_csv("data/SWaT_Dataset_Attack_v0.csv")
        pd.read_csv("data/simulator_training_data.csv")
    ]
)

for k, v in data.dtypes.items():
    print(k.ljust(8), v)

for actuator in actuators:
    data[actuator] = [int(x > 1) for x in data[actuator]]


def setup_subject(i):
    subject = data.iloc[i : i + args.timesteps + 1].copy()
    assert len(subject == args.timesteps)
    subject["id"] = i
    subject["time"] = list(range(args.timesteps + 1))
    if "Normal/Attack" in subject:
        subject["Attack"] = [x == "Attack" for x in subject["Normal/Attack"]]
    # subject = subject.loc[subject.time % 15 == 0]
    return subject


individuals = range(0, len(data) - args.timesteps, args.timeskip)
subjects = (setup_subject(i) for i in individuals)

data = pd.concat(subjects)
data.to_csv("data/long_data.csv", index=False)
