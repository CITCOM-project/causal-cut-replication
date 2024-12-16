"""
Generate all the experimental configurations.
"""

ROOT = "case-1-swat"
attacks = f"{ROOT}/successful_attacks_mutated.json"
dag = f"{ROOT}/dcg_raw.dot"
safe_ranges = f"{ROOT}/safe_ranges.json"
data = f"{ROOT}/data/data-210.pqt"
TIMESTEPS_PER_INTERVENTION = 15
TIMESTEPS = 210
attack_ids = list(range(119))

configurations = []
for sample_size in [500, 1000, 2000, 3000, 4000, 5000, None]:
    for confidence in [0.1, 0.2]:
        logs = f"{ROOT}/logs/sample_{sample_size}/ci_{round(100 - (100 * confidence))}"
        for i in attack_ids:
            n = f"-n {sample_size} " if sample_size is not None else ""
            configurations.append(
                (
                    f"-a {attacks} "
                    f"-d {dag} "
                    f"-s {safe_ranges} "
                    f"-t {TIMESTEPS_PER_INTERVENTION} "
                    f"-c {confidence} "
                    f"-i {i} "
                    f"-o {logs}/attack-{i}.json "
                    f"-T {TIMESTEPS} "
                    f"{n} -S "
                    f"{data}"
                )
            )
with open("configurations.txt", "w") as f:
    print("\n".join(configurations), file=f, end="")

print(len(configurations), "configurations")
