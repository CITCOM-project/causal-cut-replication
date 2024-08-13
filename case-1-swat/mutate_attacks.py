import json
import random

from actuators import actuators


random.seed(0)

with open("successful_attacks_flat.json") as f:
    attacks = json.load(f)

with open("usage_profile_probs.json") as f:
    usage_profile = json.load(f)

for attack in attacks:
    attack["minimal"] = [x for x in attack["attack"]]
    attack["spurious"] = []
    inx = random.choice(range(len(attack["attack"])))
    attack["spurious"].append(inx)
    # a1, v1 = ("null", 1) if inx == 0 else attack["attack"][inx - 1]
    # print((inx, len(attack["attack"])))
    # a2, v2 = ("null", 1) if inx == len(attack["attack"]) - 1 else attack["attack"][inx + 1]
    #
    # choices = usage_profile[a1][str(v1)][a2][str(v2)]
    # choices = [(variable, value, choices[variable][value]) for variable in choices for value in choices[variable]]
    # insert = random.choices(population=[(x[0], int(x[1])) for x in choices], weights=[x[2] for x in choices])
    insert = [(random.choice(actuators), random.choice([0, 1]))]
    attack["attack"].insert(inx, insert[0])

with open("successful_attacks_mutated.json", "w") as f:
    json.dump(attacks, f, indent=2)
