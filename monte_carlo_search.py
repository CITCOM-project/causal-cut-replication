from safe_ranges import safe_ranges
import pandas as pd
from statsmodels.formula.api import ols
from itertools import product
from causal_testing.specification.causal_dag import CausalDAG
import networkx as nx
from rpy2.robjects.packages import importr

data_path = "data/timestep_data.csv"
normal_path = "data/SWaT_Dataset_Normal_v0.csv"

dag_path = "dags/dag_3.dot"

data = pd.read_csv(data_path)
dag = CausalDAG(dag_path)
dagitty = importr("dagitty")
dagitty_dag = dagitty.dagitty("dag {" + ";\n".join([f"{x} -> {y}" for x, y in dag.graph.edges]) + "}")

actuators = set(
    [node.split("_")[0] for node in dag.graph.nodes if nx.get_node_attributes(dag.graph, "type")[node] == "actuator"]
)
print(actuators)


class Capability:
    def __init__(self, name, start, end, value):
        self.name = name
        self.start = start
        self.end = end
        self.value = value

    def dataframe(self):
        return pd.DataFrame([{f"{self.name}_{i}": self.value for i in range(self.start, self.end + 1)}])

    def __str__(self):
        return f"[{self.name} := {self.value}, {self.start}-{self.end}]"

    def __repr__(self):
        return f"[{self.name} := {self.value}, {self.start}-{self.end}]"


cap = 2
actuator_values = [1, 2]

for variables in product(*[actuators for _ in range(cap)]):
    for values in product(*[actuator_values for _ in range(cap)]):
        for outcome in safe_ranges:
            capabilities = [
                Capability(name=actuator, start=(15 * n), end=(15 * (n + 1)) - 1, value=value)
                for n, (actuator, value) in enumerate(zip(variables, values), 1)
            ]
            minimal_adjustment_set = set()
            for i, treatment in enumerate(variables, 1):
                [adjustment_set] = list(
                    dagitty.adjustmentSets(
                        dagitty_dag,
                        exposure=f"{variables[0]}_{i*15 - 1}",
                        outcome=f"{outcome}_{len(variables) * 15}",
                        max_results=1,
                    )
                )
                minimal_adjustment_set = minimal_adjustment_set.union(set(adjustment_set))

            features = [
                f"{capability.name}_{t}"
                for capability in capabilities
                for t in range(capability.start, capability.end + 1)
            ]
            model = ols(
                f"{outcome}_{len(capabilities) * 15} ~ {'+'.join(features+sorted(list(filter(lambda x: x not in features, minimal_adjustment_set))))}",
                data,
            ).fit()
            df = pd.concat(
                [c.dataframe() for c in capabilities],
                axis=1,
            )
            for f in minimal_adjustment_set:
                if f not in df:
                    df[f] = data.iloc[0][f]
            trace_estimate = model.predict(df)[0]
            print(
                capabilities,
                outcome,
                round(trace_estimate, 4),
                "Safe?",
                safe_ranges[outcome]["min"] < trace_estimate < safe_ranges[outcome]["max"],
            )
