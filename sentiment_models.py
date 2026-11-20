from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

import requests
try:
    from transformers import pipeline
except Exception:  # noqa: BLE001
    pipeline = None


class SentimentError(RuntimeError):
    pass


class BaseSentimentScorer:
    """Return per-text sentiment in [-1, 1]."""

    def score(self, texts: Sequence[str]) -> list[float]:
        raise NotImplementedError


@dataclass
class FinBertSentimentScorer(BaseSentimentScorer):
    model_name: str = "ProsusAI/finbert"
    device: int = -1  # -1 CPU, >=0 GPU
    batch_size: int = 16
    max_length: int = 256
    hf_endpoint: str | None = None

    def __post_init__(self) -> None:
        # Allow using HuggingFace mirror in restricted networks.
        # Common mirror: https://hf-mirror.com
        if self.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.hf_endpoint)
        if pipeline is None:
            raise SentimentError("transformers is not available")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        # When the environment has no outbound network, prevent background hub calls
        # (e.g. safetensors auto-conversion thread).
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SAFETENSORS_AUTO_CONVERSION", "1")

        # Load explicitly to ensure `use_safetensors=False` is honored and to avoid
        # transformers' auto-conversion thread requiring network access.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        # Avoid transformers' auto-conversion/network probe path here. The
        # local model cache already has weights; loading the PyTorch weights
        # directly keeps FinBERT refreshes offline and predictable.
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            use_safetensors=False,
            local_files_only=True,
            trust_remote_code=False,
        )
        self._pipe = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tok,
            device=self.device,
            truncation=True,
            max_length=self.max_length,
        )

    def score(self, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        outs = self._pipe(list(texts), batch_size=self.batch_size)
        # FinBERT labels are usually: positive / neutral / negative
        res: list[float] = []
        for o in outs:
            label = (o.get("label") or "").lower()
            p = float(o.get("score") or 0.0)
            if "pos" in label:
                res.append(p)
            elif "neg" in label:
                res.append(-p)
            else:
                res.append(0.0)
        return res


@dataclass
class OpenAICompatibleSentimentScorer(BaseSentimentScorer):
    """OpenAI-compatible Chat Completions API scorer.

    Env vars (if fields not provided):
      - OPENAI_BASE_URL (e.g. https://api.openai.com)
      - OPENAI_API_KEY
      - OPENAI_MODEL (e.g. gpt-4o-mini)
    """

    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout_s: float = 60.0
    max_retries: int = 4
    backoff_s: float = 1.0
    batch_size: int = 20

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com").rstrip(
            "/"
        )
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        self.model = self.model or os.getenv("OPENAI_MODEL")
        if not self.api_key:
            raise SentimentError("Missing OPENAI_API_KEY")
        if not self.model:
            raise SentimentError("Missing OPENAI_MODEL")
        self._session = requests.Session()

    @staticmethod
    def _extract_json(text: str) -> Any:
        # Be tolerant to markdown fences.
        t = text.strip()
        t = re.sub(r"^```(json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
        return json.loads(t)

    @staticmethod
    def _coerce_scores(arr: Any, n: int) -> list[float]:
        """Make best-effort to return exactly n floats."""
        if not isinstance(arr, list):
            return [0.0] * n
        xs = []
        for x in arr:
            try:
                xs.append(float(x))
            except Exception:
                xs.append(0.0)
        if len(xs) < n:
            xs.extend([0.0] * (n - len(xs)))
        if len(xs) > n:
            xs = xs[:n]
        return xs

    def _call(self, texts: list[str]) -> list[float]:
        url = f"{self.base_url}/v1/chat/completions"
        prompt = (
            "你是量化交易里的新闻情绪打分器。\n"
            "对每条英文财经新闻标题，判断其对美股大盘（SPY/QQQ）次日方向的情绪影响。\n"
            "输出一个 JSON 数组，长度与输入一致，每个元素是 -1 到 +1 之间的浮点数：\n"
            "- +1 强烈利好，0 中性，-1 强烈利空。\n"
            "只输出 JSON，不要解释。"
        )
        user_content = json.dumps({"titles": texts}, ensure_ascii=False)
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._session.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                if r.status_code >= 400:
                    raise SentimentError(f"OpenAI HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                arr = self._extract_json(content)
                if not isinstance(arr, list) or len(arr) != len(texts):
                    raise SentimentError(f"Bad model output: {content[:200]}")
                return [float(x) for x in arr]
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(self.backoff_s * attempt)
        raise SentimentError(f"OpenAI call failed after retries: {last_err}")

    def score(self, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        all_texts = list(texts)
        out: list[float] = []
        for i in range(0, len(all_texts), self.batch_size):
            out.extend(self._call(all_texts[i : i + self.batch_size]))
        return out


@dataclass
class ArkChatSentimentScorer(OpenAICompatibleSentimentScorer):
    """Volcengine Ark-compatible chat completions scorer.

    Defaults (override via env vars):
      - ARK_BASE_URL (default: https://ark.cn-beijing.volces.com/api/v3)
      - ARK_API_KEY
      - ARK_MODEL (endpoint id like ep-xxxx)
    """

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or os.getenv("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3").rstrip(
            "/"
        )
        self.api_key = self.api_key or os.getenv("ARK_API_KEY")
        self.model = self.model or os.getenv("ARK_MODEL")
        if not self.api_key:
            raise SentimentError("Missing ARK_API_KEY")
        if not self.model:
            raise SentimentError("Missing ARK_MODEL")
        self._session = requests.Session()

    def _call(self, texts: list[str]) -> list[float]:
        # Ark uses /chat/completions under /api/v3
        url = f"{self.base_url}/chat/completions"
        n = len(texts)
        prompt = (
            "你是量化交易里的新闻情绪打分器。\n"
            "对每条英文财经新闻标题/摘要，判断其对美股大盘（SPY/QQQ）次日方向的情绪影响。\n"
            f"输出一个 JSON 数组，长度必须为 {n}（与输入一致），每个元素是 -1 到 +1 之间的浮点数：\n"
            "- +1 强烈利好，0 中性，-1 强烈利空。\n"
            "只输出 JSON，不要解释。"
        )
        user_content = json.dumps({"texts": texts}, ensure_ascii=False)
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._session.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                if r.status_code >= 400:
                    raise SentimentError(f"ARK HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                arr = self._extract_json(content)
                xs = self._coerce_scores(arr, n)
                return xs
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(self.backoff_s * attempt)
        raise SentimentError(f"ARK call failed after retries: {last_err}")


@dataclass
class ArkStructuredSentimentScorer(ArkChatSentimentScorer):
    """Ark scorer that asks model for (score, confidence, relevance).

    Returns float per item: score * confidence * relevance.
    """

    def _call(self, texts: list[str]) -> list[float]:
        url = f"{self.base_url}/chat/completions"
        n = len(texts)
        prompt = (
            "你是量化交易里的新闻情绪打分器。\n"
            "对每条英文财经新闻标题/摘要，输出一个对象，字段：\n"
            "- score: -1 到 +1（对 SPY/QQQ 次日方向影响）\n"
            "- confidence: 0 到 1（你对判断的把握）\n"
            "- relevance: 0 到 1（与美股大盘相关性）\n"
            f"输出 JSON 数组，长度必须为 {n}，数组元素为对象；只输出 JSON。"
        )
        user_content = json.dumps({"texts": texts}, ensure_ascii=False)
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._session.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                if r.status_code >= 400:
                    raise SentimentError(f"ARK HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                arr = self._extract_json(content)
                if not isinstance(arr, list):
                    return [0.0] * n
                out = []
                for obj in arr[:n]:
                    if isinstance(obj, dict):
                        s = float(obj.get("score", 0.0))
                        c = float(obj.get("confidence", 1.0))
                        rel = float(obj.get("relevance", 1.0))
                    elif isinstance(obj, (list, tuple)) and len(obj) >= 3:
                        s, c, rel = float(obj[0]), float(obj[1]), float(obj[2])
                    else:
                        s, c, rel = 0.0, 1.0, 1.0
                    # bound
                    s = max(-1.0, min(1.0, s))
                    c = max(0.0, min(1.0, c))
                    rel = max(0.0, min(1.0, rel))
                    out.append(s * c * rel)
                if len(out) < n:
                    out.extend([0.0] * (n - len(out)))
                return out
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(self.backoff_s * attempt)
        raise SentimentError(f"ARK call failed after retries: {last_err}")


@dataclass
class GLMChatSentimentScorer(OpenAICompatibleSentimentScorer):
    """ZhipuAI GLM chat completions scorer (PaaS v4).

    Defaults (override via env vars):
      - GLM_BASE_URL (default: https://open.bigmodel.cn/api/paas/v4)
      - GLM_API_KEY
      - GLM_MODEL (e.g. glm-4-7-251222)
    """

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or os.getenv("GLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4").rstrip(
            "/"
        )
        self.api_key = self.api_key or os.getenv("GLM_API_KEY")
        self.model = self.model or os.getenv("GLM_MODEL")
        if not self.api_key:
            raise SentimentError("Missing GLM_API_KEY")
        if not self.model:
            raise SentimentError("Missing GLM_MODEL")
        self._session = requests.Session()

    def _call(self, texts: list[str]) -> list[float]:
        url = f"{self.base_url}/chat/completions"
        n = len(texts)
        prompt = (
            "你是量化交易里的新闻情绪打分器。\n"
            "对每条英文财经新闻标题/摘要，判断其对美股大盘（SPY/QQQ）次日方向的情绪影响。\n"
            f"输出一个 JSON 数组，长度必须为 {n}（与输入一致），每个元素是 -1 到 +1 之间的浮点数：\n"
            "- +1 强烈利好，0 中性，-1 强烈利空。\n"
            "只输出 JSON，不要解释。"
        )
        user_content = json.dumps({"texts": texts}, ensure_ascii=False)
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
        }
        # Zhipu uses Bearer token for API key
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._session.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                if r.status_code >= 400:
                    raise SentimentError(f"GLM HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                arr = self._extract_json(content)
                xs = self._coerce_scores(arr, n)
                return xs
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(self.backoff_s * attempt)
        raise SentimentError(f"GLM call failed after retries: {last_err}")


@dataclass
class LexiconSentimentScorer(BaseSentimentScorer):
    """Offline fallback scorer using a small finance-oriented lexicon.

    Not a true LLM; used when model downloads are unavailable.
    """

    def __post_init__(self) -> None:
        self._pos = {
            "beat",
            "beats",
            "surge",
            "surges",
            "rally",
            "rallies",
            "record high",
            "upgrade",
            "upgrades",
            "strong",
            "bullish",
            "growth",
            "profit",
            "profits",
            "optimistic",
            "cooling inflation",
            "rate cut",
            "soft landing",
        }
        self._neg = {
            "miss",
            "misses",
            "plunge",
            "plunges",
            "selloff",
            "sell-off",
            "downgrade",
            "downgrades",
            "weak",
            "bearish",
            "recession",
            "inflation surge",
            "rate hike",
            "lawsuit",
            "probe",
            "warning",
            "guidance cut",
        }

    @staticmethod
    def _norm(t: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s\-]", " ", t.lower())).strip()

    def _score_one(self, text: str) -> float:
        t = self._norm(text)
        if not t:
            return 0.0
        pos = 0
        neg = 0
        for w in self._pos:
            if w in t:
                pos += 1
        for w in self._neg:
            if w in t:
                neg += 1
        # Token-level simple cues
        tokens = t.split()
        for tok in tokens:
            if tok in {"soar", "soars", "jump", "jumps", "gain", "gains"}:
                pos += 1
            if tok in {"drop", "drops", "fall", "falls", "slump", "slumps"}:
                neg += 1
        raw = (pos - neg)
        # squash to [-1,1]
        return float(math.tanh(raw / 2.0))

    def score(self, texts: Sequence[str]) -> list[float]:
        return [self._score_one(t) for t in texts]


@dataclass
class EnsembleSentimentScorer(BaseSentimentScorer):
    """Average of multiple scorers; optional sign-only-when-agree mode.

    Used to combine FinBERT (offline ML, cheap+stable) with ark_struct
    (DeepSeek via Ark, captures macro/policy nuance). The 3-way report
    showed daily corr +0.32 — they agree often enough to be informative
    but disagree often enough to be diversifying.

    Modes:
      - mean: arithmetic mean (default)
      - agree: keep mean only when both have same sign, else 0
    """

    members: tuple = ("finbert", "ark_struct")
    mode: str = "mean"

    def __post_init__(self) -> None:
        # Allow ENSEMBLE_MEMBERS env override (comma-separated scorer names).
        env_members = os.getenv("ENSEMBLE_MEMBERS")
        if env_members:
            self.members = tuple(m.strip() for m in env_members.split(",") if m.strip())
        # Lazily build inner scorers (so failures are loud, not silent).
        self._inner = [build_scorer(name) for name in self.members]

    def score(self, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        per_scorer = [sc.score(texts) for sc in self._inner]
        out: list[float] = []
        for i in range(len(texts)):
            vals = [per_scorer[j][i] for j in range(len(self._inner))]
            mean = sum(vals) / max(len(vals), 1)
            if self.mode == "agree":
                signs = {1 if v > 0 else (-1 if v < 0 else 0) for v in vals}
                signs.discard(0)
                if len(signs) > 1:
                    out.append(0.0)
                    continue
            out.append(mean)
        return out


@dataclass
class FinGptRemoteSentimentScorer(OpenAICompatibleSentimentScorer):
    """FinGPT served behind an OpenAI-compatible endpoint (vLLM / TGI).

    Env vars (override the OpenAI defaults):
      - FINGPT_BASE_URL  e.g. http://<host>:8000  (self-hosted vLLM HTTP server)
      - FINGPT_API_KEY   any non-empty string; vLLM ignores it but the
                         OpenAICompatibleSentimentScorer base class
                         requires it.
      - FINGPT_MODEL     e.g. FinGPT/fingpt-sentiment_llama2-13b_lora
                         or whatever --served-model-name vLLM was started
                         with.
    """

    def __post_init__(self) -> None:
        # Layer FinGPT-specific env vars on top of the OpenAI defaults.
        self.base_url = (
            self.base_url or os.getenv("FINGPT_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
        ).rstrip("/")
        self.api_key = self.api_key or os.getenv("FINGPT_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        self.model = self.model or os.getenv("FINGPT_MODEL") or os.getenv("OPENAI_MODEL")
        if not self.base_url:
            raise SentimentError("Missing FINGPT_BASE_URL")
        if not self.model:
            raise SentimentError("Missing FINGPT_MODEL")
        self._session = requests.Session()


def build_scorer(kind: str) -> BaseSentimentScorer:
    kind = kind.lower().strip()
    if kind in {"finbert", "fin-bert", "bert"}:
        try:
            # Use HF_ENDPOINT if set by user.
            return FinBertSentimentScorer(hf_endpoint=os.getenv("HF_ENDPOINT"))
        except Exception:
            return LexiconSentimentScorer()
    if kind in {"openai", "llm", "chat"}:
        return OpenAICompatibleSentimentScorer()
    if kind in {"ark", "doubao", "deepseek"}:
        return ArkChatSentimentScorer()
    if kind in {"ark_struct", "doubao_struct", "deepseek_struct"}:
        return ArkStructuredSentimentScorer()
    if kind in {"glm", "zhipu"}:
        return GLMChatSentimentScorer()
    if kind in {"fingpt", "fin-gpt", "fingpt_remote"}:
        return FinGptRemoteSentimentScorer()
    if kind in {"ensemble", "ensemble_mean"}:
        return EnsembleSentimentScorer(mode="mean")
    if kind in {"ensemble_agree", "ensemble_strict"}:
        return EnsembleSentimentScorer(mode="agree")
    if kind in {"lex", "lexicon", "rule"}:
        return LexiconSentimentScorer()
    raise ValueError(f"Unknown scorer kind: {kind}")
