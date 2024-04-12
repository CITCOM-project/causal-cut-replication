#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:28:31 2024

@author: michael
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.generation.abstract_causal_test_case import AbstractCausalTestCase
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Meta, Output
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_result import CausalTestResult
from causal_testing.testing.estimators import Estimator, LinearRegressionEstimator, LogisticRegressionEstimator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_test_adequacy import DataAdequacy
from causal_testing.testing.causal_test_outcome import SomeEffect

from rpy2.robjects.packages import importr

from config import *

start = time.time()


df = pd.read_csv(data_path, index_col=0)

causal_dag = CausalDAG(dag_path)

dtypes = {np.dtype("float64"): float, np.dtype("int64"): int}

variables = []
for node, data in causal_dag.graph.nodes(data=True):
    variable = Input
    if data["type"] == "actuator":
        variable = Output
    variables.append(variable(node, dtypes[df.dtypes[node]] if node in df else float))

scenario = Scenario(variables)
scenario.setup_treatment_variables()
causal_specification = CausalSpecification(scenario, causal_dag)
data_collector = ObservationalDataCollector(scenario, data)

treatment = "MV101_1"
outcome = f"{outcome}_{timesteps}"


base_test_case = BaseTestCase(
    treatment_variable=scenario.variables[treatment],
    outcome_variable=scenario.variables[outcome],
    effect="total",
)
causal_test_case = CausalTestCase(
    base_test_case=base_test_case,
    expected_causal_effect=SomeEffect,
    estimate_type="coefficient",
    effect_modifier_configuration={},
)


estimator_kwargs = {}
print("Identifying")
dagitty = importr("dagitty")
dagitty_dag = dagitty.dagitty("dag {" + ";\n".join([f"{x} -> {y}" for x, y in causal_dag.graph.edges]) + "}")
[adjustment_set] = list(dagitty.adjustmentSets(dagitty_dag, exposure=treatment, outcome=outcome, max_results=1))
minimal_adjustment_set = set(adjustment_set)
# minimal_adjustment_set = causal_specification.causal_dag.identification(causal_test_case.base_test_case, max_results=1)
print("Identified", minimal_adjustment_set)
estimator_kwargs["adjustment_set"] = minimal_adjustment_set

estimator_kwargs["treatment"] = causal_test_case.treatment_variable.name
estimator_kwargs["treatment_value"] = causal_test_case.treatment_value
estimator_kwargs["control_value"] = causal_test_case.control_value
estimator_kwargs["outcome"] = causal_test_case.outcome_variable.name
estimator_kwargs["effect_modifiers"] = causal_test_case.effect_modifier_configuration
estimator_kwargs["alpha"] = 0.05

normal_data = df.query("not Attack")
assert len(normal_data) > 0
normal_estimation_estimator = LinearRegressionEstimator(**estimator_kwargs | {"df": normal_data})
normal_causal_test_result = causal_test_case.execute_test(
    estimator=normal_estimation_estimator, data_collector=data_collector
)
print(normal_causal_test_result)

attack_data = df.query("Attack")
assert len(attack_data) > 0
attack_estimation_estimator = LinearRegressionEstimator(**estimator_kwargs | {"df": attack_data})
attack_causal_test_result = causal_test_case.execute_test(
    estimator=attack_estimation_estimator, data_collector=data_collector
)
print(attack_causal_test_result)

prediction_model = LinearRegressionEstimator(**estimator_kwargs | {"df": df}).model().fit()


end = time.time()
print(end - start)

fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
ax1.scatter(normal_data[treatment], normal_data[outcome])
ax2.scatter(attack_data[treatment], attack_data[outcome])
ax1.set_title("Normal")
ax2.set_title("Attack")
plt.show()
