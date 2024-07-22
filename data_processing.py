import json

with open("successful_attacks.json") as f:
    successful_attacks = json.load(f)

with open("logs/output_no_filter_lo_hi_sim_results.json") as f:
    results = json.load(f)

timesteps_per_intervention = 15

outcomes = 0
attack_strategies = 0
events = 0
estimated_without_error = 0
estimated_significant = 0
adequate = 0
inadequate = 0

counted = []

for outcome, attacks in successful_attacks.items():
    outcomes += 1
    outcome, attack = outcome.split(" ")
    for capabilities in attacks:
        attack_strategies += 1
        events += len(capabilities)
        control_strategy = [
            {"variable": var, "value": val, "start_time": t, "end_time": t + timesteps_per_intervention}
            for (var, val), t in zip(
                capabilities,
                range(
                    timesteps_per_intervention,
                    (len(capabilities) * timesteps_per_intervention) + 1,
                    timesteps_per_intervention,
                ),
            )
        ]
        treatment_strategies = [
            datum for datum in results if datum["outcome"] == outcome and datum["control_strategy"] == control_strategy
        ]

        for treatment_strategy in treatment_strategies:
            assert treatment_strategy not in counted
            counted.append(treatment_strategy)
            estimated_without_error += 1
            significant = int(treatment_strategy["significant"])
            estimated_significant += significant
            threshold = 0.1
            if significant:
                adequate += int(-threshold < treatment_strategy["kurtosis"] < threshold)
            if not significant:
                inadequate += int(-threshold < treatment_strategy["kurtosis"] < threshold)


print(f"{outcomes} outcomes")
print(f"{attack_strategies} attacks consisting of {events} events")
print(f"{estimated_without_error} estimated without error")
print(f"{estimated_significant} estimated significant, {adequate} estimated adequate")
print(f"{events - estimated_significant} estimated insignificant, {inadequate} estimated adequate")
