"""Dependency-free adapter from resolved world config to the Rust API."""
from __future__ import annotations


def family_from_config(params: dict):
    from world_py import FamilyParams

    return FamilyParams(
        n_hyp=params["n_hyp"],
        n_probe=params["n_probe"],
        n_evidence=params["n_evidence"],
        cost_lo=params["cost_lo"],
        cost_hi=params["cost_hi"],
        budget_slack=params["budget_slack"],
        min_depth=params["min_depth"],
        step_slack=params["step_slack"],
        variant=params["variant"],
    )
