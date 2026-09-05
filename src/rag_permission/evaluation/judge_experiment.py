import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from rag_permission.citations import is_refusal
from rag_permission.evaluation.judges import (
    FAITHFULNESS_SYSTEM,
    RELEVANCE_SYSTEM,
    extract_json,
)
from rag_permission.llm import LLMClient


@dataclass(frozen=True, slots=True)
class JudgePromptVariant:
    name: str
    faithfulness_system: str
    relevance_system: str


@dataclass(frozen=True, slots=True)
class JudgeCase:
    query: str
    answer: str
    passages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JudgePromptScores:
    prompt: str
    faithfulness_samples: tuple[float, ...]
    relevance_samples: tuple[float, ...]
    faithfulness_mean: float
    relevance_mean: float
    faithfulness_stdev: float
    relevance_stdev: float


RAGAS_STYLE = JudgePromptVariant(
    name="ragas_style",
    faithfulness_system=FAITHFULNESS_SYSTEM,
    relevance_system=RELEVANCE_SYSTEM,
)
GROUNDING_STRICT = JudgePromptVariant(
    name="grounding_strict",
    faithfulness_system="""You are a strict grounding judge.
Split the answer into atomic claims. Mark a claim supported only when a numbered
source explicitly states every fact in the claim. Treat inference, interpolation,
and partial support as unsupported. Respond with JSON:
{"assertions":[{"text":"...","supported":true}]}
""",
    relevance_system="""You are a strict answer relevance judge.
Compare the answer with the question and ignore citation formatting. Respond with
exactly one label: relevant, partially_relevant, or irrelevant.
""",
)
DEFAULT_PROMPTS = (RAGAS_STYLE, GROUNDING_STRICT)


def _faithfulness_score(
    llm: LLMClient,
    prompt_variant: JudgePromptVariant,
    case: JudgeCase,
    temperature: float,
) -> float:
    if is_refusal(case.answer):
        return 1.0
    source_text = "\n\n".join(
        f"[{number}] {passage}" for number, passage in enumerate(case.passages, 1)
    )
    response = llm.complete(
        prompt_variant.faithfulness_system,
        f"Question: {case.query}\nAnswer: {case.answer}\nSources:\n{source_text}",
        temperature,
    )
    assertions = extract_json(response.text).get("assertions", [])
    if not assertions:
        return 0.0
    return sum(bool(item.get("supported")) for item in assertions) / len(assertions)


def _relevance_score(
    llm: LLMClient,
    prompt_variant: JudgePromptVariant,
    case: JudgeCase,
    temperature: float,
) -> float:
    response = llm.complete(
        prompt_variant.relevance_system,
        f"Question: {case.query}\nAnswer: {case.answer}",
        temperature,
    )
    label = response.text.strip().lower()
    return {"relevant": 1.0, "partially_relevant": 0.5, "irrelevant": 0.0}.get(
        label.split()[0] if label else "", 0.0
    )


def _summarize_samples(prompt: str, faith: list[float], relevance: list[float]) -> JudgePromptScores:
    def stdev(values: list[float]) -> float:
        return statistics.stdev(values) if len(values) > 1 else 0.0

    return JudgePromptScores(
        prompt=prompt,
        faithfulness_samples=tuple(faith),
        relevance_samples=tuple(relevance),
        faithfulness_mean=statistics.mean(faith) if faith else 0.0,
        relevance_mean=statistics.mean(relevance) if relevance else 0.0,
        faithfulness_stdev=stdev(faith),
        relevance_stdev=stdev(relevance),
    )


class JudgeExperiment:
    def __init__(
        self,
        llm: LLMClient,
        prompt_variants: tuple[JudgePromptVariant, ...] = DEFAULT_PROMPTS,
        samples: int = 5,
        temperature: float = 0.2,
    ):
        if samples < 2:
            raise ValueError("samples must be at least 2")
        self.llm = llm
        self.prompt_variants = prompt_variants
        self.samples = samples
        self.temperature = temperature

    def run(self, cases: list[JudgeCase]) -> dict:
        prompt_results: dict[str, dict] = {}
        for prompt_variant in self.prompt_variants:
            case_scores = []
            for case in cases:
                faith = [
                    _faithfulness_score(self.llm, prompt_variant, case, self.temperature)
                    for _ in range(self.samples)
                ]
                relevance = [
                    _relevance_score(self.llm, prompt_variant, case, self.temperature)
                    for _ in range(self.samples)
                ]
                summary = _summarize_samples(prompt_variant.name, faith, relevance)
                case_scores.append({"query": case.query, "scores": summary})
            prompt_results[prompt_variant.name] = {
                "cases": case_scores,
                "faithfulness_mean": statistics.mean(
                    item["scores"].faithfulness_mean for item in case_scores
                ),
                "relevance_mean": statistics.mean(
                    item["scores"].relevance_mean for item in case_scores
                ),
                "mean_faithfulness_stdev": statistics.mean(
                    item["scores"].faithfulness_stdev for item in case_scores
                ),
                "mean_relevance_stdev": statistics.mean(
                    item["scores"].relevance_stdev for item in case_scores
                ),
            }
        names = [item.name for item in self.prompt_variants]
        drift = {
            "faithfulness": abs(
                prompt_results[names[0]]["faithfulness_mean"]
                - prompt_results[names[1]]["faithfulness_mean"]
            ),
            "relevance": abs(
                prompt_results[names[0]]["relevance_mean"]
                - prompt_results[names[1]]["relevance_mean"]
            ),
        }
        return {"prompts": prompt_results, "prompt_drift": drift}


def load_judge_cases(path: Path | str) -> list[JudgeCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        JudgeCase(item["query"], item["answer"], tuple(item["passages"]))
        for item in data["cases"]
    ]
