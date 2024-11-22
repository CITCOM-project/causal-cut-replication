"""
This module mutates the pre-minimised attacks by adding in additional, (hopefully) spurious interventions.
"""

import json
from scipy.stats import beta
import numpy as np
from copy import deepcopy

from actuators import actuators


np.random.seed(0)

with open("successful_attacks.json") as f:
    attacks = json.load(f)

mutations_dist = beta(1.831704249184929, 56806698550007.91, -0.48454197283949485, 124788476392356.78)

new_attacks = []

attack_id = 0
for attack in attacks:
    attack["minimal"] = attack["attack"]
    attack["attack_id"] = attack_id
    attack_id += 1
    new_attacks.append(attack)
    for mutations in mutations_dist.rvs(size=3):
        attack = deepcopy(attack)
        attack["attack_id"] = attack_id
        attack_id += 1
        mutations = round(mutations)
        attack["attack"] = list(attack["minimal"])
        for _ in range(mutations):
            inx = int(np.random.choice(range(len(attack["attack"]) + 1)))
            attack["attack"].insert(inx, (None, np.random.choice(actuators), int(np.random.choice([0, 1]))))
        new_times = {old_t: t * 15 for t, (old_t, _, _) in enumerate(attack["attack"], 1) if old_t is not None}
        attack["attack"] = [(t * 15, var, val) for t, (_, var, val) in enumerate(attack["attack"], 1)]
        attack["minimal"] = [(new_times[t], var, val) for (t, var, val) in attack["minimal"]]
        attack["spurious"] = [i for i, x in enumerate(attack["attack"]) if x not in attack["minimal"]]
        new_attacks.append(attack)

with open("successful_attacks_mutated.json", "w") as f:
    json.dump(new_attacks, f, indent=2)

max_attack_length = max(len(attack["attack"]) for attack in new_attacks)
print("Maximum attack length is", max_attack_length)
print("That corresponds to", (max_attack_length + 2) * 15, "timesteps")
