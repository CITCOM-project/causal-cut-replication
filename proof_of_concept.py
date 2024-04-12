import pandas as pd
from statsmodels.formula.api import ols

data_path = "data/timestep_data.csv"
data = pd.read_csv(data_path)

attack_path = "data/SWaT_Dataset_Attack_v0.csv"
attack_df = pd.read_csv(attack_path)
attack_df["Timestamp"] = pd.to_datetime(attack_df["Timestamp"], format="%d/%m/%Y %I:%M:%S %p")


class Capability:
    def __init__(self, name, start, end, value):
        self.name = name
        self.start = start
        self.end = end
        self.value = value


# Show that MV101 is causal
capabilities = [Capability(name="MV101", start=14, end=29, value=2)]
outcome = {"name": "FIT101", "time": 30, "safe": lambda x: 0 <= x < 2}

# Get the attack trace
attack_start = pd.to_datetime("28/12/2015 10:29:14 AM", format="%d/%m/%Y %I:%M:%S %p")
trace_start = attack_start - pd.to_timedelta(15, unit="s")
trace_end = attack_start + pd.to_timedelta(15, unit="s")
trace = attack_df.loc[(attack_df["Timestamp"] >= trace_start) & (attack_df["Timestamp"] <= trace_end)]
print(trace)

# Train the model on the data
features = [
    f"{capability.name}_{t}" for capability in capabilities for t in range(capability.start, capability.end + 1)
]
model = ols(f"{outcome['name']}_{outcome['time']} ~ {'+'.join(features)}", data).fit()


print("Estimates for capability MV101")
trace_estimate = model.predict(pd.DataFrame([{k: 2 for k in features}]))[0]
counterfactual_estimate = model.predict(pd.DataFrame([{k: 1 for k in features}]))[0]
print("  Valve closed(?):", round(trace_estimate, 4), "Safe?", outcome["safe"](trace_estimate))
print("  Valve open(?):  ", round(counterfactual_estimate, 4), "Safe?", outcome["safe"](counterfactual_estimate))
print()

# Show that a spurious capability is non causal
capabilities = [
    Capability(name="MV201", start=14, end=29, value=2),
    Capability(name="MV101", start=30, end=44, value=2),
]
outcome = {"name": "FIT101", "time": 45, "safe": lambda x: 0 <= x < 2}

# Get the attack trace
attack_start = pd.to_datetime("28/12/2015 10:29:14 AM", format="%d/%m/%Y %I:%M:%S %p")
trace_start = attack_start - pd.to_timedelta(30, unit="s")
trace_end = attack_start + pd.to_timedelta(15, unit="s")
trace = attack_df.loc[(attack_df["Timestamp"] >= trace_start) & (attack_df["Timestamp"] <= trace_end)]

# Train the model on the data
features = [
    f"{capability.name}_{t}" for capability in capabilities for t in range(capability.start, capability.end + 1)
]
adjustment_set = [
    "AIT201_2",
    "AIT201_3",
    "AIT202_1",
    "AIT203_1",
    "AIT401_6",
    "AIT402_6",
    "AIT501_6",
    "AIT501_7",
    "AIT502_6",
    "AIT502_7",
    "AIT503_6",
    "AIT503_7",
    "AIT504_4",
    "AIT504_5",
    "DPIT301_2",
    "DPIT301_3",
    "FIT101_6",
    "FIT201_2",
    "FIT201_3",
    "FIT301_2",
    "FIT301_3",
    "FIT401_6",
    "FIT501_6",
    "FIT501_7",
    "FIT502_4",
    "FIT502_5",
    "FIT503_4",
    "FIT503_5",
    "LIT101_4",
    "LIT101_5",
    "LIT301_1",
    "LIT401_4",
    "LIT401_5",
    "MV101_5",
    "MV101_6",
    "MV301_3",
    "MV301_4",
    "MV302_3",
    "MV302_4",
    "P101_3",
    "P101_4",
    "P102_3",
    "P102_4",
    "P201_1",
    "P201_2",
    "P202_1",
    "P202_2",
    "P203_1",
    "P203_2",
    "P204_1",
    "P204_2",
    "P205_1",
    "P205_2",
    "P206_1",
    "P206_2",
    "P301_1",
    "P301_2",
    "P302_1",
    "P302_2",
    "P401_5",
    "P401_6",
    "P402_5",
    "P402_6",
    "P403_7",
    "P404_7",
    "P501_5",
    "P501_6",
    "P502_5",
    "P502_6",
    "P602_3",
    "P602_4",
]
model = ols(f"{outcome['name']}_{outcome['time']} ~ {'+'.join(features+adjustment_set)}", data).fit()


def get(k, trace, capability, value):
    name, time = k.split("_")
    if name == capability.name:
        return value
    return trace[name].iloc[int(time)]


for cap in capabilities:
    print(f"Estimates for capability", cap.name, cap.value)
    estimate_open = model.predict(
        pd.DataFrame([{k: get(k, trace, cap, cap.value) for k in features + adjustment_set}])
    )[0]
    print("  Valve open(?):", round(estimate_open, 4), "Safe?", outcome["safe"](estimate_open))
    estimate_closed = model.predict(pd.DataFrame([{k: get(k, trace, cap, 1) for k in features + adjustment_set}]))[0]
    print("  Valve closed(?):", round(estimate_closed, 4), "Safe?", outcome["safe"](estimate_closed))
