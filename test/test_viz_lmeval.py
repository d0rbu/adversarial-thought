from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from viz.lmeval import average_score, extract_task_scores

if TYPE_CHECKING:
    from pathlib import Path


def test_extract_task_scores_prefers_exact_task_over_subtasks(tmp_path: Path) -> None:
    # If task key exists, it should be used (not averaged with subtasks).
    data = {
        "evaluation_summary": {"tasks": ["mmlu", "xquad"]},
        "results": {
            "mmlu": {"acc": 0.7},
            "mmlu_subject": {"acc": 0.1},  # should be ignored in favor of "mmlu"
            "xquad_en": {"f1": 0.5},
            "xquad_es": {"f1": 0.7},
        },
    }
    p = tmp_path / "eval.yaml"
    p.write_text(yaml.safe_dump(data))

    scores = extract_task_scores(p)
    assert scores["mmlu"] == 0.7
    assert scores["xquad"] == (0.5 + 0.7) / 2


def test_average_score(tmp_path: Path) -> None:
    data = {
        "evaluation_summary": {"tasks": ["hellaswag", "winogrande"]},
        "results": {
            "hellaswag": {"acc_norm": 0.8},
            "winogrande": {"acc": 0.6},
        },
    }
    p = tmp_path / "eval.yaml"
    p.write_text(yaml.safe_dump(data))

    scores = extract_task_scores(p)
    assert average_score(scores) == (0.8 + 0.6) / 2
