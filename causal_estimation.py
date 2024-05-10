#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:28:31 2024

@author: michael
"""

import pandas as pd
import numpy as np

from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.generation.abstract_causal_test_case import AbstractCausalTestCase
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.testing.causal_test_result import CausalTestResult
from causal_testing.testing.estimators import Estimator, LinearRegressionEstimator, LogisticRegressionEstimator
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.causal_test_adequacy import DataAdequacy
from causal_testing.testing.causal_test_outcome import SomeEffect

from rpy2.robjects.packages import importr

from config import *
from safe_ranges import safe_ranges

data_path = data_path.replace(".csv", "_end.csv")

df = pd.read_csv(data_path, index_col=0)

within_safe_range = np.ones(len(df)).astype(bool)

dag_path = "dags/dag_3_reduced.dot"
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

treatment_var = "MV101"
start = 15
end = 30
# Can we collapse these in the DAG?
treatment = f"{treatment_var}_{start}"
# outcome = f"{outcome}_{timesteps}"
outcome = "LIT101_60"
print("Outcome", outcome)


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
[adjustment_set] = list(
    dagitty.adjustmentSets(dagitty_dag, exposure=treatment, outcome=outcome, max_results=1, effect="total")
)
minimal_adjustment_set = set(adjustment_set)
print("R Identified", minimal_adjustment_set)


minimal_adjustment_set = causal_dag.identification(causal_test_case.base_test_case, max_results=10)

parents = set(
    filter(
        lambda n: int(n.split("_")[1]) == start - timesteps_per_intervention,
        causal_dag.graph.nodes,
    )
)

minimal_adjustment_set = parents

print("Identified", minimal_adjustment_set)
estimator_kwargs["adjustment_set"] = minimal_adjustment_set

estimator_kwargs["treatment"] = causal_test_case.treatment_variable.name
estimator_kwargs["treatment_value"] = causal_test_case.treatment_value
estimator_kwargs["control_value"] = causal_test_case.control_value
estimator_kwargs["outcome"] = causal_test_case.outcome_variable.name
estimator_kwargs["effect_modifiers"] = causal_test_case.effect_modifier_configuration
estimator_kwargs["alpha"] = 0.05

normal_data = df.query("not attack")

prediction_model = LinearRegressionEstimator(**estimator_kwargs | {"df": df})._run_linear_regression()
print(prediction_model.params)
normal_trace = normal_data.iloc[[4]].copy()
print("    Observed value", normal_trace[outcome])
prediction = prediction_model.predict(normal_trace)
print("    Normal Prediction", prediction[4])
normal_trace[treatment] = 2
prediction = prediction_model.predict(normal_trace)
print("    Attack Prediction", prediction[4])
