import json

from rag_permission.evaluation.judge_experiment import (
    JudgeCase,
    JudgeExperiment,
    load_judge_cases,
)
from rag_permission.llm import LLMResponse


class DeterministicJudge:
    def __init__(self, supported: bool = True, relevance: str = "relevant"):
        self.supported = supported
        self.relevance = relevance
        self.calls = 0

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
        self.calls += 1
        if "assertions" in system.lower():
            payload = {"assertions": [{"text": "claim", "supported": self.supported}]}
            return LLMResponse(json.dumps(payload))
        return LLMResponse(self.relevance)

    def stream(self, system: str, user: str, temperature: float = 0.0):
        raise AssertionError("judge experiment should not stream")


def test_judge_experiment_samples_multiple_prompts():
    judge = DeterministicJudge()
    result = JudgeExperiment(judge, samples=3).run(
        [JudgeCase("q", "answer", ("source",))]
    )
    assert judge.calls == 12
    assert result["prompts"]["ragas_style"]["faithfulness_mean"] == 1.0
    assert result["prompt_drift"] == {"faithfulness": 0.0, "relevance": 0.0}


def test_judge_experiment_detects_prompt_drift():
    class PromptSensitiveJudge(DeterministicJudge):
        def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
            if "strict grounding" in system.lower():
                payload = {"assertions": [{"text": "claim", "supported": False}]}
                return LLMResponse(json.dumps(payload))
            return super().complete(system, user, temperature)

    result = JudgeExperiment(PromptSensitiveJudge(), samples=2).run(
        [JudgeCase("q", "answer", ("source",))]
    )
    assert result["prompt_drift"]["faithfulness"] == 1.0


def test_load_judge_cases():
    cases = load_judge_cases("fixtures/judge_experiment_set.json")
    assert len(cases) == 4
    assert cases[0].passages[0].startswith("故障码")
