import pygraphviz as pgv
import sys

dag1 = pgv.AGraph(sys.argv[1])
dag2 = pgv.AGraph(sys.argv[2])

for edge in dag1.edges():
    if edge not in dag2.edges():
        print(f"Missing edge: {edge[0]} -> {edge[1]}")
for edge in dag2.edges():
    if edge not in dag1.edges():
        print(f"Spurious edge: {edge[0]} -> {edge[1]}")
