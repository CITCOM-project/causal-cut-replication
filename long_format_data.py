import pandas as pd
import config

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

data = pd.concat([pd.read_csv("data/SWaT_Dataset_Normal_v0.csv"), pd.read_csv("data/SWaT_Dataset_Attack_v0.csv")])

for actuator in actuators:
    data[actuator] = [int(x > 1) for x in data[actuator]]


def setup_subject(i):
    subject = data.iloc[i : i + config.timesteps + 1].copy()
    assert len(subject == config.timesteps)
    subject["id"] = i
    subject["time"] = list(range(config.timesteps + 1))
    subject["Attack"] = [x == "Attack" for x in subject["Normal/Attack"]]
    subject = subject.loc[subject.time % 15 == 0]
    return subject


individuals = range(0, len(data) - config.timesteps, config.timesteps)
subjects = (setup_subject(i) for i in individuals)

data = pd.concat(subjects)
data.to_csv("data/long_data.csv", index=False)
