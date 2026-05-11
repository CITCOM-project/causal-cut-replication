"""
This module provides constants for colours and names.
"""

RED = "#D55E00"
BLUE = "#0072B2"
GREEN = "#009E73"
MAGENTA = "#CC79A7"

TOOLNAME = "CausalCut"
BASELINE = "Greedy Heuristic"
GOLD_STANDARD = "Gold Standard"
DDMIN = "DDmin"

RANGE_1 = color_hex_codes = ["#E69F00", "#56B4E9", "#CC79A7", "#AFAFAF"]
# RANGE_1 = color_hex_codes = [GREEN] * 4 + [MAGENTA] * 4

OUTCOMES = {
    "_cost_efficiency": ["original_length", "minimal", "estimable_per_event"],  # RQ1
    "": ["original_length", "minimal", "sample_size"],  # RQ2
    "_executions": ["original_length", "minimal", "sample_size"],  # RQ3
}
y_labels = {
    "_cost_efficiency": "Cost efficiency",
    "": "Reduced test length",
    "_executions": "Executions",
    "_reinstatement": "Reinstatement rate",
}
FEATURES = ["original_length", "minimal", "sample_size", "ci_alpha", "estimable_per_event"]
x_labels = {
    "original_length": "Original test length",
    "minimal": "Proportion of necessary interventions",
    "sample_size": "Executions available",
    "estimable_per_event": "Proportion of estimable interventions",
    "ci_alpha": "CI alpha",
}
technique_labels_latex = {
    "greedy_heuristic": "\\greedy",
    "ddmin": "\\ddmin",
    "causal_cut": "\\toolname",
    "reinstated": "\\toolname Phase 2 reinstated",
    "causal_cut_plus_greedy_heuristic": "\\toolnamePlus",
    "estimated_interventions": f"\\toolname Phase 1",
}
technique_labels_plain = {
    "greedy_heuristic": BASELINE,
    "ddmin": "DDmin",
    "causal_cut": TOOLNAME,
    "causal_cut_plus_greedy_heuristic": f"{TOOLNAME} + {BASELINE}",
    "estimated_interventions": f"{TOOLNAME} Phase 1",
}
technique_colours = {
    "greedy_heuristic": RED,
    "ddmin": BLUE,
    "causal_cut": GREEN,
    "causal_cut_plus_greedy_heuristic": MAGENTA,
    "estimated_interventions": "orange",
}
technique_markers = {
    "greedy_heuristic": "x",
    "ddmin": "^",
    "causal_cut": "o",
    "causal_cut_plus_greedy_heuristic": "+",
    "estimated_interventions": "*",
}
