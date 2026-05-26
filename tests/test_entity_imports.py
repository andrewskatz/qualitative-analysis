import importlib
import sys


def _reset_entity_modules():
    prefixes = [
        "qualitative_analysis.entity",
        "qualitative_analysis.entity.consolidator",
        "qualitative_analysis.entity.visualizer",
        "qualitative_analysis.entity.comparison_viz",
        "qualitative_analysis.entity.bayesian",
    ]
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            sys.modules.pop(name, None)


def test_entity_package_import_does_not_eagerly_load_heavy_modules():
    _reset_entity_modules()

    pkg = importlib.import_module("qualitative_analysis.entity")

    assert pkg.EntityScorer is not None
    assert "qualitative_analysis.entity.consolidator" not in sys.modules
    assert "qualitative_analysis.entity.visualizer" not in sys.modules
    assert "qualitative_analysis.entity.comparison_viz" not in sys.modules
    assert "qualitative_analysis.entity.bayesian" not in sys.modules


def test_entity_package_lazy_loads_consolidator_on_access():
    _reset_entity_modules()

    pkg = importlib.import_module("qualitative_analysis.entity")

    assert "qualitative_analysis.entity.consolidator" not in sys.modules
    assert pkg.EntityConsolidator is not None
    assert "qualitative_analysis.entity.consolidator" in sys.modules


def test_entity_package_lazy_loads_visualizer_on_access():
    _reset_entity_modules()

    pkg = importlib.import_module("qualitative_analysis.entity")

    assert "qualitative_analysis.entity.visualizer" not in sys.modules
    assert pkg.EntityVisualizer is not None
    assert "qualitative_analysis.entity.visualizer" in sys.modules
