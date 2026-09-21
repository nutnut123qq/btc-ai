from research_contract import DEFAULT_EXECUTION_COSTS, ResearchManifest


def test_execution_cost_units_are_explicit_and_consistent():
    costs = DEFAULT_EXECUTION_COSTS
    assert costs.total_per_side_bps == 15.0
    assert costs.round_trip_bps == 30.0
    assert costs.round_trip_pct_points == 0.30
    assert costs.round_trip_return_fraction == 0.003


def test_manifest_hash_is_stable_and_changes_with_parameters():
    contract = ResearchManifest(experiment="fixture")
    first = contract.to_dict(parameters={"window": 15}, data_provenance={"hash": "abc"})
    repeated = contract.to_dict(parameters={"window": 15}, data_provenance={"hash": "abc"})
    changed = contract.to_dict(parameters={"window": 20}, data_provenance={"hash": "abc"})
    assert first["manifestSha256"] == repeated["manifestSha256"]
    assert first["manifestSha256"] != changed["manifestSha256"]
    assert first["signalBarState"] == "closed"


def test_manifest_hash_includes_code_and_runtime_provenance_when_supplied():
    contract = ResearchManifest(experiment="fixture")
    first = contract.to_dict(
        parameters={},
        data_provenance={"hash": "abc"},
        code_provenance={"gitCommit": "deadbeef", "implementationContentSha256": "one"},
        runtime_dependencies={"python": "3.12.0"},
    )
    changed = contract.to_dict(
        parameters={},
        data_provenance={"hash": "abc"},
        code_provenance={"gitCommit": "deadbeef", "implementationContentSha256": "two"},
        runtime_dependencies={"python": "3.12.0"},
    )
    assert first["manifestSha256"] != changed["manifestSha256"]
