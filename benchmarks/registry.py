"""Registry mapping --dataset to gold files, prompts, and payload builders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.prompts import (
    APPROVER_LINE_SYSTEM_PROMPT as TAX_APPROVER_LINE,
    FILER_LINE_SYSTEM_PROMPT as TAX_FILER_LINE,
    FILER_SYSTEM_PROMPT as TAX_FILER,
    VERIFIER_LINE_SYSTEM_PROMPT as TAX_VERIFIER_LINE,
    VERIFIER_SYSTEM_PROMPT as TAX_VERIFIER,
)
from core.agents import _trim_input_for_verifier
from benchmarks.convfinqa import cfq_prompts
from benchmarks.tax import tax_linewise

_LLM_ROOT = Path(__file__).resolve().parent.parent
_BENCH_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    gold_filename: str
    form_name: str
    return_version: str
    default_cases_root: Path
    default_eval_cases_root: Path
    filer_line_system_prompt: str
    verifier_line_system_prompt: str
    approver_line_system_prompt: str
    filer_system_prompt: str
    verifier_system_prompt: str
    line_filer_user_payload: Callable[..., str]
    line_verifier_user_payload: Callable[..., str]
    line_revise_user_payload: Callable[..., str]
    line_approver_user_payload: Callable[..., str]
    trim_input_for_verifier: Callable[..., Any]


DATASETS: Dict[str, DatasetConfig] = {
    "tax": DatasetConfig(
        name="tax",
        gold_filename="output.xml",
        form_name="1040",
        return_version="ty24-v1",
        default_cases_root=_LLM_ROOT / "tax-calc-bench" / "tax_calc_bench" / "ty24" / "test_data",
        default_eval_cases_root=_LLM_ROOT / "tax-calc-bench" / "tax_calc_bench" / "ty24" / "test_data",
        filer_line_system_prompt=TAX_FILER_LINE,
        verifier_line_system_prompt=TAX_VERIFIER_LINE,
        approver_line_system_prompt=TAX_APPROVER_LINE,
        filer_system_prompt=TAX_FILER,
        verifier_system_prompt=TAX_VERIFIER,
        line_filer_user_payload=tax_linewise.line_filer_user_payload,
        line_verifier_user_payload=tax_linewise.line_verifier_user_payload,
        line_revise_user_payload=tax_linewise.line_revise_user_payload,
        line_approver_user_payload=tax_linewise.line_approver_user_payload,
        trim_input_for_verifier=_trim_input_for_verifier,
    ),
    "finqa": DatasetConfig(
        name="finqa",
        gold_filename="gold.json",
        form_name=cfq_prompts.FORM_FINQA,
        return_version="finqa-v1",
        default_cases_root=_BENCH_ROOT / "convfinqa" / "data" / "finqa" / "train",
        default_eval_cases_root=_BENCH_ROOT / "convfinqa" / "data" / "finqa" / "test",
        filer_line_system_prompt=cfq_prompts.FILER_LINE_SYSTEM_PROMPT,
        verifier_line_system_prompt=cfq_prompts.VERIFIER_LINE_SYSTEM_PROMPT,
        approver_line_system_prompt=cfq_prompts.APPROVER_LINE_SYSTEM_PROMPT,
        filer_system_prompt=cfq_prompts.FILER_LINE_SYSTEM_PROMPT,
        verifier_system_prompt=cfq_prompts.VERIFIER_LINE_SYSTEM_PROMPT,
        line_filer_user_payload=cfq_prompts.line_filer_user_payload,
        line_verifier_user_payload=cfq_prompts.line_verifier_user_payload,
        line_revise_user_payload=cfq_prompts.line_revise_user_payload,
        line_approver_user_payload=cfq_prompts.line_approver_user_payload,
        trim_input_for_verifier=cfq_prompts.trim_input_for_verifier,
    ),
    "convfinqa": DatasetConfig(
        name="convfinqa",
        gold_filename="gold.json",
        form_name=cfq_prompts.FORM_CONVFINQA,
        return_version="convfinqa-v1",
        default_cases_root=_BENCH_ROOT / "convfinqa" / "data" / "convfinqa" / "train",
        default_eval_cases_root=_BENCH_ROOT / "convfinqa" / "data" / "convfinqa" / "test",
        filer_line_system_prompt=cfq_prompts.FILER_LINE_SYSTEM_PROMPT,
        verifier_line_system_prompt=cfq_prompts.VERIFIER_LINE_SYSTEM_PROMPT,
        approver_line_system_prompt=cfq_prompts.APPROVER_LINE_SYSTEM_PROMPT,
        filer_system_prompt=cfq_prompts.FILER_LINE_SYSTEM_PROMPT,
        verifier_system_prompt=cfq_prompts.VERIFIER_LINE_SYSTEM_PROMPT,
        line_filer_user_payload=cfq_prompts.line_filer_user_payload,
        line_verifier_user_payload=cfq_prompts.line_verifier_user_payload,
        line_revise_user_payload=cfq_prompts.line_revise_user_payload,
        line_approver_user_payload=cfq_prompts.line_approver_user_payload,
        trim_input_for_verifier=cfq_prompts.trim_input_for_verifier,
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}")
    return DATASETS[name]


def gold_path_for_case(case_dir: Path, dataset: str) -> Path:
    return case_dir / get_dataset_config(dataset).gold_filename
