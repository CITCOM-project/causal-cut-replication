import json
import random

from actuators import actuators

random.seed(0)

with open("successful_attacks_flat.json") as f:
    attacks = json.load(f)

for attack in attacks:
    attack["minimal"] = [x for x in attack["attack"]]
    inx = random.choice(range(len(attack["attack"]) + 1))
    attack["spurious"] = [inx]
    attack["attack"].insert(inx, (random.choice(actuators), random.choice([1, 0])))

with open("successful_attacks_mutated.json", "w") as f:
    json.dump(attacks, f, indent=2)
