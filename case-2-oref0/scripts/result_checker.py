"""
This module checks the estimated sequences of attacks to see which ones still yield a fault.
"""

import json
from sys import argv
from multiprocessing import Pool
import pandas as pd

if __name__ == "__main__":
    assert len(argv) == 2, "Please provide a JSON results file to process."
    with open(argv[1]) as f:
        attacks = json.load(f)

    with Pool() as pool:
        processed_attacks = pool.map(build_attack, sorted(attacks, key=lambda a: a["attack_index"]))
    with open(argv[1].replace(".json", "_reproduced.json"), "w") as f:
        json.dump(processed_attacks, f)

    results = [
        {
            "attack_length": len(attack["attack"]),
            "estimated_success": attack["pure_estimate_fault"],
            "estimated_length": len(attack["estimated_interventions"]),
            "estimated_spurious": len(
                [(t, var) for (t, var, _) in attack["estimated_interventions"] if [t, var, 1] not in attack["minimal"]]
            ),
            "extended_success": attack["extended_estimate_fault"],
            "extended_length": len(attack["extended_interventions"]),
            "extended_spurious": len(
                [(t, var) for (t, var, _) in attack["extended_interventions"] if [t, var, 1] not in attack["minimal"]]
            ),
        }
        for attack in processed_attacks
    ]

    for attack in processed_attacks:
        if (
            len(
                set(map(lambda x: tuple(x[:2]), attack["minimal"])).difference(
                    set(map(lambda x: tuple(x[:2]), attack["extended_interventions"]))
                )
            )
            > 0
        ):
            print("MINIMAL", attack["minimal"])
            print("EXTENDED", sorted(attack["extended_interventions"]))

    results = pd.DataFrame(results)
    results["estimated_success"] = results["estimated_success"].astype(bool)
    results["extended_success"] = results["extended_success"].astype(bool)

    results.to_csv("/home/michael/tmp/results.csv")

    results["estimated_proportions"] = results["estimated_length"] / results["attack_length"]
    results["extended_proportions"] = results["extended_length"] / results["attack_length"]

    print("=" * 40)
    print(f"{results['estimated_success'].sum()}/{len(attacks)} successful attacks")
    print(
        "Successful estimated attacks were overall "
        f"{(results.loc[results['estimated_success'], 'estimated_proportions'].mean()*100).round(3)}% "
        "of the original attack"
    )
    print(
        "Successful estimated attacks contained on average "
        f"{results['estimated_spurious'].mean().round(3)} spurious events"
    )
    print(
        "Successful estimated attacks contained on average "
        f"{((results['estimated_spurious']/results['estimated_length']).mean()*100).round(3)}% spurious events"
    )
    print("=" * 40)
    print(f"{results['extended_success'].sum()}/{len(attacks)} successful extended attacks")
    print(
        "Successful extended attacks were overall "
        f"{(results.loc[results['extended_success'], 'extended_proportions'].mean()*100).round(3)}% "
        "of the original attack"
    )
    print(
        "Successful extended attacks contained on average "
        f"{results['extended_spurious'].mean().round(3)} spurious events"
    )
    print(
        "Successful extended attacks contained on average "
        f"{((results['extended_spurious']/results['extended_length']).mean()*100).round(3)}% spurious events"
    )
    print("=" * 40)
    print(
        "Extended attacks contained on average "
        f"{(results['extended_length'] - results['estimated_length']).mean().round(3)} "
        "events more than the pure estimated"
    )
