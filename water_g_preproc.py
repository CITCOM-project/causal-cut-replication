import pandas as pd
import config
from collections import OrderedDict
from safe_ranges import safe_ranges
import random

df = pd.read_csv("data/long_data.csv")
df["trtrand"] = None  # treatment/control arm
df["fault_t_do"] = None  # did a fault occur here?
df["xo_t_do"] = None  # did the individual deviate from the treatment of interest here?
df["now_prog_t_dc"] = None  # has the situation progressed now?
df["recent_prog_t_dc"] = None  # has the situation progressed in the past?
df["fault_time"] = None  # when did a fault occur?

control = config.interventions
treatment = control.copy()
treatment["P101"] = 1

control_strategy = list(
    zip(
        control,
        range(config.timesteps_per_intervention, config.timesteps + 1, config.timesteps_per_intervention),
        control.values(),
    )
)
treatment_strategy = list(
    zip(
        treatment,
        range(config.timesteps_per_intervention, config.timesteps + 1, config.timesteps_per_intervention),
        treatment.values(),
    )
)


def setup_xo_t_do(strategy_assigned, strategy_followed):
    censored = False
    result = []
    for x, y in zip(strategy_assigned, strategy_followed):
        if censored:
            result.append(None)
        else:
            result.append((not censored) and x != y)
            censored = x != y
    # First and last can't be censored
    return [0] + [int(x) if x is not None else None for x in result] + [0 if result[-1] is not None else None]


def setup_fault_t_do(values, min, max):
    fault_occurred = False
    result = []
    for value in values:
        fault = not (min <= value <= max)
        result.append((not fault_occurred) and fault)
        if (not fault_occurred) and fault:
            fault_occurred == True
    return [int(x) for x in result]


def setup_recent_prog_t_dc(now_prog_t_dc):
    result = []
    prog = False
    for value in now_prog_t_dc:
        prog = prog or value
        result.append(prog)
    return result


individuals = []
new_id = 0
for id, individual in df.groupby("id"):
    strategy = [
        (var, time, individual.loc[individual.time == time, var].values[0]) for var, time, _ in treatment_strategy
    ]
    individual["fault_t_do"] = setup_fault_t_do(
        individual[config.outcome], safe_ranges[config.outcome]["lolo"], safe_ranges[config.outcome]["hihi"]
    )
    individual["now_prog_t_dc"] = setup_fault_t_do(
        individual[config.outcome], safe_ranges[config.outcome]["lo"], safe_ranges[config.outcome]["hi"]
    )
    individual["recent_prog_t_dc"] = setup_recent_prog_t_dc(individual["now_prog_t_dc"])
    faulty = individual.loc[
        ~individual[config.outcome].between(safe_ranges[config.outcome]["lolo"], safe_ranges[config.outcome]["hihi"]),
        "time",
    ]
    fault_time = individual.time.max()
    if len(faulty) > 0:
        fault_time = faulty.min()  # TODO: update this to take values from the full dataset
    individual["fault_time"] = random.randint(fault_time + 1, fault_time + 15)

    if fault_time <= 0:
        continue

    # Control flow:
    # Individuals that start off in both arms, need cloning (hence incrementing the ID within the if statements)
    # Individuals that don't start off in either arm need leaving out (hence two ifs rather than elif or else)
    if strategy[0] == control_strategy[0]:
        individual["id"] = id
        id += 1
        individual["trtrand"] = 0
        individual["xo_t_do"] = setup_xo_t_do(strategy, control_strategy)
        individuals.append(individual.loc[individual.time <= fault_time].copy())
    if strategy[0] == treatment_strategy[0]:
        individual["id"] = id
        id += 1
        individual["trtrand"] = 1
        individual["xo_t_do"] = setup_xo_t_do(strategy, treatment_strategy)
        individuals.append(individual.loc[individual.time <= fault_time].copy())
df = pd.concat(individuals)
df.to_csv("data/long_preprocessed.csv")
