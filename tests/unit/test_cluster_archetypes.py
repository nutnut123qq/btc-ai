from cluster_archetypes import primary_horizon_for


def test_primary_horizon_matches_the_labels_used_for_clustering():
    assert primary_horizon_for("1h") == "4h"
    assert primary_horizon_for("4h") == "4h"
    assert primary_horizon_for("1d") == "1d"
