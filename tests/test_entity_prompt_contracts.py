from qualitative_analysis.entity.models import DimensionDefinition
from qualitative_analysis.entity.scorer import EntityScorer


def _build_prompt(version: str) -> str:
    scorer = EntityScorer(prompt_version=version)
    return scorer._build_prompt(
        entity="renewable energy",
        context="The community invested in renewable energy infrastructure.",
        dimensions=[
            DimensionDefinition(name="Social", description="Social factors"),
            DimensionDefinition(name="Ecological", description="Ecological factors"),
            DimensionDefinition(name="Technological", description="Technological factors"),
        ],
    )


def test_v2_prompt_contract_includes_reasoning_and_score_then_justification():
    prompt = _build_prompt("v2")
    first_dimension = prompt.index('"dimension": "[dimension_name_1]"')
    score_idx = prompt.index('"score":', first_dimension)
    justification_idx = prompt.index('"justification":', first_dimension)

    assert '"initial_observations":' in prompt
    assert 'Include brief justifications that reference the context' in prompt
    assert score_idx < justification_idx


def test_v3_prompt_contract_includes_reasoning_and_justification_before_score():
    prompt = _build_prompt("v3")
    first_dimension = prompt.index('"dimension": "[dimension_name_1]"')
    justification_idx = prompt.index('"justification":', first_dimension)
    score_idx = prompt.index('"score":', first_dimension)

    assert '"initial_observations":' in prompt
    assert 'the "justification" field MUST come BEFORE the "score" field' in prompt
    assert justification_idx < score_idx


def test_v4_prompt_contract_is_score_only():
    prompt = _build_prompt("v4")

    assert '"dimension_scores":' in prompt
    assert '"initial_observations":' not in prompt
    assert '"justification":' not in prompt
    assert 'Return ONLY the JSON object, no additional text' in prompt


def test_prompt_versions_remain_distinct_workflow_modes():
    prompt_v2 = _build_prompt("v2")
    prompt_v3 = _build_prompt("v3")
    prompt_v4 = _build_prompt("v4")

    assert 'Now analyze the following entity:' in prompt_v2
    assert 'Now analyze the following entity:' in prompt_v3
    assert 'Now score the following entity:' in prompt_v4
    assert prompt_v2 != prompt_v3 != prompt_v4