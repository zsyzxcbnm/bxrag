"""LLM-as-a-Judge evaluation for context relevance and faithfulness."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.config import Settings
from src.exceptions import ConfigurationError
from src.models import QueryExample, SearchHit
from src.retrieval.pipeline import RetrievalPipeline

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class ChatJudge(Protocol):
    """Minimal text-generation interface used by the judge runner."""

    name: str

    def generate(self, prompt: str) -> str:
        """Return the model response text."""


class LocalChatJudge:
    """Local Hugging Face chat model, suitable for offline evaluation."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 256,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ConfigurationError(
                "transformers/torch are required for the local judge"
            ) from exc

        self.name = model_path
        self._torch = torch
        self._tokenizer: Any = AutoTokenizer.from_pretrained(model_path)
        self._device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model: Any = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )
        self._model.to(self._device)  # type: ignore[arg-type]
        self._model.eval()
        self._max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        rendered = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        return str(
            self._tokenizer.decode(
                output[0][prompt_length:],
                skip_special_tokens=True,
            )
        ).strip()


class OpenAIJudge:
    """Optional OpenAI-backed judge for higher-quality offline-free evaluation."""

    def __init__(self, model: str, api_key: str | None) -> None:
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for the OpenAI judge")
        from openai import OpenAI

        self.name = model
        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return (response.choices[0].message.content or "").strip()


def build_judge(settings: Settings) -> ChatJudge:
    """Build the configured judge provider."""

    if settings.judge_provider == "openai":
        return OpenAIJudge(settings.judge_model, settings.openai_api_key)
    return LocalChatJudge(
        settings.judge_model,
        device=settings.judge_device or "cpu",
        max_new_tokens=settings.judge_max_new_tokens,
    )


def _extract_json(text: str) -> dict[str, Any]:
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError("Judge response did not contain a JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Judge response JSON must be an object")
    return payload


def _clamp_score(value: object) -> int:
    try:
        score = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("Judge score must be an integer") from exc
    return max(1, min(5, score))


def _context_text(hits: list[SearchHit], max_chars: int = 700) -> str:
    return "\n\n".join(
        f"[{hit.doc_id}] {hit.title}\n{hit.content[:max_chars]}" for hit in hits
    )


def _judge_context(
    judge: ChatJudge,
    query: str,
    hits: list[SearchHit],
) -> dict[str, Any]:
    prompt = f"""You are a strict retrieval evaluator for a bank support assistant.
Rate whether the retrieved passages contain enough information to answer the
question. Use this scale:
5 = directly and completely answers the question
4 = mostly answers, minor detail missing
3 = partially answers
2 = weak or ambiguous
1 = irrelevant

Question:
{query}

Retrieved passages:
{_context_text(hits)}

Return ONLY a JSON object:
{{"score": <integer 1-5>, "reason": "<short reason>"}}
"""
    raw = judge.generate(prompt)
    try:
        payload = _extract_json(raw)
        return {
            "score": _clamp_score(payload.get("score")),
            "reason": str(payload.get("reason", ""))[:300],
            "raw": raw,
            "parse_error": False,
        }
    except (ValueError, json.JSONDecodeError) as exc:
        return {"score": 0, "reason": str(exc), "raw": raw, "parse_error": True}


def _generate_answer(judge: ChatJudge, query: str, hits: list[SearchHit]) -> str:
    prompt = f"""Answer the customer-support question using ONLY the retrieved
policy passages. Cite each factual claim with its [DOC-ID]. If the passages do
not contain enough evidence, say "insufficient evidence". Keep the answer to at
most three sentences.

Question:
{query}

Passages:
{_context_text(hits)}

Answer (one short sentence; include the exact number and unit when asked):
"""
    return judge.generate(prompt)


def _judge_faithfulness(
    judge: ChatJudge,
    query: str,
    answer: str,
    hits: list[SearchHit],
) -> dict[str, Any]:
    prompt = f"""You are a strict factuality evaluator. Check whether every factual
claim in the answer is supported by the retrieved passages. Do not use outside
knowledge.

Question:
{query}

Answer:
{answer}

Passages:
{_context_text(hits)}

Score:
5 = every claim is supported
4 = mostly supported, minor wording issue
3 = some unsupported detail
2 = substantial unsupported content
1 = answer contradicts or ignores the passages

Return ONLY a JSON object:
{{"score": <integer 1-5>, "unsupported_claims": ["..."], "reason": "<short reason>"}}
"""
    raw = judge.generate(prompt)
    try:
        payload = _extract_json(raw)
        unsupported = payload.get("unsupported_claims", [])
        if not isinstance(unsupported, list):
            unsupported = [str(unsupported)]
        return {
            "score": _clamp_score(payload.get("score")),
            "unsupported_claims": [str(item)[:200] for item in unsupported[:5]],
            "reason": str(payload.get("reason", ""))[:300],
            "raw": raw,
            "parse_error": False,
        }
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "score": 0,
            "unsupported_claims": [],
            "reason": str(exc),
            "raw": raw,
            "parse_error": True,
        }


@dataclass(frozen=True, slots=True)
class JudgeSummary:
    context_relevance: float
    faithfulness: float
    parse_errors: int
    queries: int


def run_judge(
    settings: Settings,
    pipeline: RetrievalPipeline,
    queries: list[QueryExample],
    judge: ChatJudge,
    top_k: int = 5,
    limit: int | None = None,
) -> tuple[JudgeSummary, list[dict[str, Any]]]:
    """Run retrieval, answer generation and both judge prompts."""

    selected = queries[:limit] if limit is not None else queries
    rows: list[dict[str, Any]] = []
    context_scores: list[int] = []
    faithfulness_scores: list[int] = []
    parse_errors = 0

    for query in selected:
        started = time.perf_counter()
        result = pipeline.search(query.query, mode="hybrid", top_k=top_k, use_metadata=False)
        context_eval = _judge_context(judge, query.query, result.hits)
        answer = _generate_answer(judge, query.query, result.hits)
        faithfulness_eval = _judge_faithfulness(judge, query.query, answer, result.hits)
        elapsed = (time.perf_counter() - started) * 1000

        if not context_eval["parse_error"]:
            context_scores.append(int(context_eval["score"]))
        else:
            parse_errors += 1
        if not faithfulness_eval["parse_error"]:
            faithfulness_scores.append(int(faithfulness_eval["score"]))
        else:
            parse_errors += 1

        rows.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "retrieved_doc_ids": [hit.doc_id for hit in result.hits],
                "context_relevance": context_eval,
                "answer": answer,
                "faithfulness": faithfulness_eval,
                "elapsed_ms": round(elapsed, 3),
            }
        )

    summary = JudgeSummary(
        context_relevance=(
            sum(context_scores) / len(context_scores) if context_scores else 0.0
        ),
        faithfulness=(
            sum(faithfulness_scores) / len(faithfulness_scores)
            if faithfulness_scores
            else 0.0
        ),
        parse_errors=parse_errors,
        queries=len(selected),
    )
    return summary, rows


def write_judge_report(
    summary: JudgeSummary,
    rows: list[dict[str, Any]],
    reports_dir: Path,
    judge_name: str,
) -> Path:
    """Write judge results as JSON and Markdown."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "judge_model": judge_name,
        "summary": {
            "queries": summary.queries,
            "context_relevance_mean_1_to_5": round(summary.context_relevance, 3),
            "faithfulness_mean_1_to_5": round(summary.faithfulness, 3),
            "parse_errors": summary.parse_errors,
        },
        "rows": rows,
    }
    (reports_dir / "llm_judge.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "| Query | Context relevance | Faithfulness | Retrieved docs |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        context = row["context_relevance"]
        faith = row["faithfulness"]
        lines.append(
            f"| {row['query_id']} | {context['score'] if not context['parse_error'] else 'ERR'} | "
            f"{faith['score'] if not faith['parse_error'] else 'ERR'} | "
            f"{', '.join(row['retrieved_doc_ids'])} |"
        )
    table = "\n".join(lines)
    report = f"""# LLM-as-a-Judge Evaluation

Judge model: `{judge_name}`

## Summary

- Queries: **{summary.queries}**
- Context relevance (1-5): **{summary.context_relevance:.3f}**
- Faithfulness (1-5): **{summary.faithfulness:.3f}**
- JSON parse errors: **{summary.parse_errors}**

## Per-query results

{table}

## Method

For each query, the evaluator:

1. retrieves the top-5 passages with the rules-free hybrid RRF pipeline;
2. asks the judge model to rate whether the passages answer the question;
3. asks the judge model to answer using only those passages;
4. asks the judge model whether every claim in that answer is supported.

The judge is intentionally separate from the retrieval metrics and is not used
to tune the retriever.
"""
    path = reports_dir / "llm_judge.md"
    path.write_text(report, encoding="utf-8")
    return path

