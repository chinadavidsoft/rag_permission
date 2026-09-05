import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag_permission.config import Settings
from rag_permission.evaluation.judge_experiment import JudgeExperiment, load_judge_cases
from rag_permission.llm import OpenAILLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare judge prompts with repeated sampling")
    parser.add_argument("--cases", default="fixtures/judge_experiment_set.json")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output")
    args = parser.parse_args()
    settings = Settings()
    llm = OpenAILLMClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    result = JudgeExperiment(llm, samples=args.samples, temperature=args.temperature).run(
        load_judge_cases(args.cases)
    )
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=asdict) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result["prompt_drift"], ensure_ascii=False))
    for name, scores in result["prompts"].items():
        print(
            name,
            f"faithfulness={scores['faithfulness_mean']:.3f}",
            f"relevance={scores['relevance_mean']:.3f}",
            f"stdev={scores['mean_faithfulness_stdev']:.3f}/{scores['mean_relevance_stdev']:.3f}",
        )


if __name__ == "__main__":
    main()
