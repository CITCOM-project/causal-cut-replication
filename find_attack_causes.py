import pandas as pd
import json
from rpy2.robjects.packages import importr
from statsmodels.formula.api import ols

from causal_testing.specification.causal_dag import CausalDAG

from safe_ranges import safe_ranges
from config import timesteps_per_intervention

attack_path = "data/SWaT_Dataset_Attack_v0.csv"
attack_df = pd.read_csv(attack_path)

training_data_path = "data/timestep_data.csv"
training_data = pd.read_csv(training_data_path)


with open("successful_attacks.json") as f:
    attacks = json.load(f)
# attacks = {
#     tuple((tuple(x) for x in sequence)): {"sensor": sensor, "causes": []}
#     for sensor, sequences in attacks.items()
#     for sequence in sequences
# }

dag_path = "dags/dag_3.dot"
causal_dag = CausalDAG(dag_path)
dagitty = importr("dagitty")
dagitty_dag = dagitty.dagitty("dag {" + ";\n".join([f"{x} -> {y}" for x, y in causal_dag.graph.edges]) + "}")

to_num = {"off": 1, "on": 2, "close": 1, "open": 2}
negate = {1: 2, 2: 1}

attack_causes = {}

for inx, g in attack_df.groupby(attack_df["attack"].ne(attack_df["attack"].shift()).cumsum()):
    if not g["attack"].all():
        continue
    print("Attack", int(inx / 2))

    for k in attacks:
        sensor, end = k.split(" ")
        print(sensor, end)
        for sequence in attacks[k]:
            print(sequence)
            data = attack_df.iloc[
                g.index[0] - timesteps_per_intervention : g.index[0] + (timesteps_per_intervention * len(sequence)) + 1
            ].reset_index(drop=True)
            print(
                "Considering data indices",
                g.index[0] - timesteps_per_intervention,
                "to",
                g.index[0] + (timesteps_per_intervention * len(sequence)) + 1,
            )
            # data = g.iloc[: timesteps_per_intervention * (len(sequence) + 1)].reset_index(drop=True)
            nodes = {x.split("_")[0] for x in causal_dag.graph.nodes()}
            if (
                sensor not in safe_ranges
                or len(sequence) > 3
                or any(act not in data for act, _ in sequence)
                or any(act not in nodes for act, _ in sequence)
            ):
                print(" ", sensor, "invalid")
                continue

            if "High" in end:
                print(f"{sensor} High and not {data[sensor].iloc[-1]} < {safe_ranges[sensor]['max']}")
            else:
                print("  skipping - not a high test")
                continue

            if ("Low" in end and not safe_ranges[sensor]["min"] <= data[sensor].iloc[-1]) or (
                "High" in end and not data[sensor].iloc[-1] < safe_ranges[sensor]["max"]
            ):
                print(f"  Sensor {sensor} reading {data[sensor].iloc[-1]} is too {end[1:-1]}")
            else:
                print(f"  Sensor {sensor} reading {data[sensor].iloc[-1]} is safe")
                continue

            minimal_adjustment_set = set()
            outcome = f"{sensor}_{(len(sequence)+1) * timesteps_per_intervention}"
            for i, treatment in enumerate(sequence, 1):
                for t in range(timesteps_per_intervention):
                    exposure = f"{treatment[0]}_{(i*timesteps_per_intervention)+t}"
                    [adjustment_set] = list(
                        dagitty.adjustmentSets(
                            dagitty_dag,
                            exposure=exposure,
                            outcome=outcome,
                            effect="total",
                            max_results=1,
                        )
                    )
                    minimal_adjustment_set = minimal_adjustment_set.union(set(adjustment_set))

            features = [
                f"{capability}_{t}"
                for i, (capability, _) in enumerate(sequence, 1)
                for t in range(i * timesteps_per_intervention, (i + 1) * timesteps_per_intervention)
            ]
            data.pop("Timestamp")
            data.pop("Normal/Attack")
            data.pop("attack")
            df = data.stack()
            df.index = df.index.map("{0[1]}_{0[0]}".format)
            df = df.to_frame().T

            model = ols(
                f"{outcome} ~ {'+'.join(features+sorted(list(filter(lambda x: x not in features, minimal_adjustment_set))))}",
                training_data,
            ).fit()

            causal_caps = []
            print("   ", sequence)
            for i, (act, val) in enumerate(sequence, 1):
                dfc = df.copy()
                for inx in range(i * timesteps_per_intervention, (i + 1) * timesteps_per_intervention):
                    dfc[f"{act}_{inx}"] = negate[to_num[val]]
                trace_estimate = model.predict(dfc)[0]

                print(
                    "      ",
                    act,
                    val,
                    sensor,
                    round(trace_estimate, 4),
                    "Safe?",
                    safe_ranges[sensor]["min"] < trace_estimate < safe_ranges[sensor]["max"],
                )

                print("Causal effects", model.conf_int())
                print(model.summary())

                causal_caps.append((act, val, safe_ranges[sensor]["min"] < trace_estimate < safe_ranges[sensor]["max"]))
            print("   ", causal_caps)
            if all(x[2] for x in causal_caps):
                print("FOUND ONE!")
                if sequence not in attack_causes:
                    attack_causes[sequence] = []
                attack_causes[sequence].append(inx)
            assert False
    break

for attack in attack_causes:
    print(attack, attack_causes[attack])
# print(
#     sequence,
#     sensor,
#     round(trace_estimate, 4),
#     "Safe?",
#     safe_ranges[sensor]["min"] < trace_estimate < safe_ranges[sensor]["max"],
# )
