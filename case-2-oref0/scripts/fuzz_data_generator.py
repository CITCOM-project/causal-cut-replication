"""
This module generates attack sequences by fuzzing the successful attacks.
This simulates the way a fuzzer would work, searching for successful attacks by
mutating traces.
"""

import json
from multiprocessing import Pool
import random
import numpy as np
from scipy.stats import binom
import argparse

from abstract_data_generator import DataGenerator
from aps_digitaltwin.util import intervention_values, beta_iob, beta_cob

from dotenv import load_dotenv


class FuzzDataGenerator(DataGenerator):
    """
    Generate a dataset by fuzzing the attack trace by adding and removing random interventions.
    """

    def __init__(self, max_steps, root, resamples):
        self.max_steps = max_steps
        self.root = root
        self.resamples = resamples
        self.covered = set()

    def add_intervention(self, attack: list):
        """
        Mutate an attack trace by adding one intervention at a random time point
        that does not already have an intervention.

        :param attack: The attack to mutate.
        """
        times = [t for t, _, _ in attack]
        time = random.choice(sorted(list(set(range(self.max_steps)).difference(times))))
        var, val = random.choice(sorted(list(intervention_values.items())))
        attack.append((time, var, val))

    def remove_intervention(self, attack: list):
        """
        Mutate an attack trace by removing one intervention.

        :param attack: The attack to mutate.
        """
        if len(attack) > 0:
            attack.pop(random.randint(0, len(attack) - 1))

    def generate_attacks(self, attack: dict):
        """
        Generate attacks by randomly adding and removing around 20% of interventions.

        :param attack: A dictionary containing the constants and interventions that consistitute the attack.
        """
        yield (
            attack["attack_id"],
            "unmodified",
            "original",
            attack["constants"],
            [(t, v, intervention_values[v]) for t, v, _ in attack["attack"]],
            attack["initial_bg"],
            attack["initial_carbs"],
            attack["initial_iob"],
        )

        dist = binom(len(attack["attack"]), 0.1)
        for r, mutations in enumerate(dist.rvs(size=self.resamples).astype(int)):
            seed = attack["attack_id"] + r
            mutations = max(mutations, 1)

            to_mutate = [(t, v, intervention_values[v]) for t, v, _ in attack["attack"]]
            for _ in range(mutations):
                f = np.random.choice((self.add_intervention, self.remove_intervention), p=[0.4, 0.6])
                f(to_mutate)

            yield (
                attack["attack_id"],
                "fuzzed",
                r,
                self.random_constants(),
                to_mutate,
                self.safe_initial_bg(seed),
                beta_cob.rvs(1, random_state=seed)[0],
                beta_iob.rvs(1, random_state=seed)[0],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Fuzz data generator")
    parser.add_argument(
        "-r",
        "--resamples",
        type=int,
        default=500,
        help="The number of test sequences to generate from each attack. Defaults to 500",
    )
    parser.add_argument(
        "-S",
        "--systematic",
        action="store_true",
        default=False,
        help="Systematically generate the same number of test sequences per attack as CtrlTrtDataGenerator."
        "Defaults to False to generate the same number of sequences for each attack."
        "Note that, if enabled, this will generate len(attack) * resamples traces per attack.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=1,
        help="The random seed.",
    )
    args = parser.parse_args()

    load_dotenv()
    random.seed(args.seed)
    np.random.seed(args.seed)

    generator = FuzzDataGenerator(max_steps=500, root="data-fuzz", resamples=500)

    THREADS = 15
    LOSS_RATE = 0.01

    with open("successful_attacks.json") as filepath:
        attacks = json.load(filepath)

    with Pool(THREADS) as pool:
        for i, a in enumerate(attacks):
            print(f"Attack {a['attack_id']} ({i+1} of {len(attacks)})")
            if args.systamtic:
                c2 = LOSS_RATE * args.resamples
                c1 = c2 + args.resamples
                generator.resamples = (c2 * (len(a) ** 2)) + (c1 * len(a)) + args.resamples
            pool.map(generator.one_iteration, generator.generate_attacks(a))
