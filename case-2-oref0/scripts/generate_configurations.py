"""
Generate all the experimental configurations.
"""

ROOT = "case-2-oref0"
attacks = f"{ROOT}/successful_attacks.json"
dag = f"{ROOT}/dcg.dot"
safe_ranges = f"{ROOT}/safe_ranges.json"
TIMESTEPS_PER_INTERVENTION = 5
TIMESTEPS = 500
BASELINE_CONFOUNDERS = "kjs kgj kjl kgl kxg kxgi kxi τ η kλ kμ Gprod0"
attack_ids = [
    2,
    10,
    14,
    16,
    18,
    19,
    35,
    42,
    45,
    56,
    57,
    59,
    68,
    70,
    84,
    85,
    105,
    112,
    116,
    118,
    126,
    129,
    130,
    131,
    137,
    138,
    139,
    145,
    157,
    163,
    164,
    169,
    176,
    177,
    182,
    183,
    195,
    215,
    219,
    220,
    244,
    276,
    280,
    285,
    289,
    292,
    309,
    311,
    333,
    337,
    339,
    361,
    363,
    364,
    379,
    381,
    382,
    390,
    392,
    395,
    401,
    430,
    445,
    448,
    451,
    456,
    460,
    463,
    469,
    477,
]

configurations = []
for dataset in range(5):
    for sample_size in [50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000]:
        for confidence in [0.1, 0.2]:
            logs = f"{ROOT}/logs/fuzzed_attacks_{dataset}/sample_{sample_size}/ci_{round(100 - (100 * confidence))}"
            for i in attack_ids:
                data = f"{ROOT}/data/fuzz_data_{dataset}_attacks/{i}.pqt"
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
                        f"-b {BASELINE_CONFOUNDERS} "
                        f"-n {sample_size} "
                        "-S "
                        f"{data}"
                    )
                )
with open("configurations.txt", "w") as f:
    print("\n".join(configurations), file=f, end="")

print(len(configurations), "configurations")
