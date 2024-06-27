import pygraphviz as pgv
from itertools import product
from multiprocessing import Pool
import networkx as nx
import pandas as pd

from config import outcome, interventions, timesteps_per_intervention, timesteps, data_path

flow = pgv.AGraph("flow.dot")
flow.write("flow_raw.dot")
data = pd.read_csv(data_path, index_col=0)

# Validate that I've drawn the flow graph correctly
for node1 in flow.nodes():
    # Sensors ONLY affect (and are affected by) actuators
    if node1.attr["type"] == "sensor":
        assert all(node2.attr["type"] == "actuator" for node2 in flow.in_neighbors(node1) + flow.successors(node1))
    # Actuators ONLY affect (and are affected by) sensors
    elif node1.attr["type"] == "actuator":
        assert all(node2.attr["type"] == "sensor" for node2 in flow.in_neighbors(node1) + flow.successors(node1))
    else:
        raise TypeError("Node type must be 'sensor' or 'actuator'")


# --- Build the DAG by unfolding timesteps
dag = pgv.AGraph(directed=True)
memo = {}


# Memoisation for SPEED!
def add_edge(memo, cause, effect):
    if (cause, effect) in memo:
        return memo[(cause, effect)]
    elif (effect, cause) in memo:
        return memo[(effect, cause)]
    else:
        memo[(cause, effect)] = (cause, effect) in flow.edges() or (effect, cause) in flow.edges()
        return memo[(cause, effect)]


# Unfold the timesteps
for t in range(timesteps):
    for cause in flow.nodes():
        dag.add_node(f"{cause}_{t}", type=cause.attr["type"])
        dag.add_node(f"{cause}_{t+1}", type=cause.attr["type"])
        if cause.attr["type"] == "sensor":
            dag.add_edge(f"{cause}_{t}", f"{cause}_{t+1}")
        for effect in flow.nodes():
            if add_edge(memo, cause, effect):
                dag.add_node(f"{effect}_{t+1}", type=effect.attr["type"])
                dag.add_edge(f"{cause}_{t}", f"{effect}_{t+1}")

dag = nx.nx_agraph.from_agraph(dag)
nx.drawing.nx_agraph.write_dot(dag, f"dags/dag_{len(interventions)}.dot")
# ---

# Prune the DAG by removing nodes with no path to the outcome
# nodes = list(dag.nodes)
# for node in nodes:
#     if not nx.has_path(dag, node, f"{outcome}_{timesteps}"):
#         dag.remove_node(node)

# Set the interventions by removing incoming edges
# for t, intervention in enumerate(interventions, 1):
#     t *= timesteps_per_intervention
#     dag.remove_edges_from([(node, f"{intervention}_{t}") for node in dag.predecessors(f"{intervention}_{t}")])
#     nx.set_node_attributes(dag, {f"{intervention}_{t}": {"fillcolor": "red", "style": "filled"}})

# print(data.columns)
# nodes_to_delete = [node for node in dag.nodes if node not in data.columns]
# for node in nodes_to_delete:
#     dag.remove_node(node)

nx.drawing.nx_agraph.write_dot(dag, f"dags/dag_{len(interventions)}.dot")
