import json
import os
from typing import Any


class OpenAICompatibleJudge:
    """Adapt an OpenAI-compatible chat client to the evaluator's judge protocol."""

    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def __call__(
        self,
        case: dict[str, str],
        trajectory: list[dict[str, Any]],
        answer: str,
    ) -> dict[str, Any]:
        prompt = {
            "case": case,
            "trajectory": trajectory,
            "answer": answer,
            "requirements": {
                "score": "number from 0 to 1",
                "passed": "boolean; true when the answer fulfills the case",
                "reason": "short explanation",
            },
        }
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Evaluate the research answer. Return only valid JSON with score, passed, and reason.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
            ],
        )
        content = response.choices[0].message.content
        try:
            result = json.loads(content)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("LLM judge must return valid JSON") from error
        if not isinstance(result, dict) or "score" not in result or "passed" not in result:
            raise ValueError("LLM judge JSON must contain score and passed")
        return {
            "score": float(result["score"]),
            "passed": bool(result["passed"]),
            "reason": str(result.get("reason", "")),
            "source": f"llm:{self.model}",
        }


def build_llm_judge_from_env() -> OpenAICompatibleJudge:
    """Build an OpenAI-compatible judge from environment variables.

    NVIDIA_API_KEY selects NVIDIA NIM defaults; OPENAI_* variables remain
    supported for other OpenAI-compatible providers.
    """
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY or OPENAI_API_KEY is required when --judge llm is selected")

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the optional openai package to use --judge llm") from error

    base_url = os.getenv("NVIDIA_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if os.getenv("NVIDIA_API_KEY") and not base_url:
        base_url = "https://integrate.api.nvidia.com/v1"
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    model = os.getenv("NVIDIA_MODEL") or os.getenv("OPENAI_MODEL")
    return OpenAICompatibleJudge(client, model or "z-ai/glm-5.3")