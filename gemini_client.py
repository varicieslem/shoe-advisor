"""
gemini_client.py
================
Gemini API (yeni nesil `google-genai` SDK) için tek noktadan erişim katmanı.

- API anahtarı: Colab Secrets (`GEMINI_API_KEY`), yoksa ortam değişkeni.
- Embedding: `embed_content` (batch + retry + L2 normalizasyon).
- Üretim: düz metin ve JSON Mode (response_schema ile yapılandırılmış çıktı).
- Model fallback: ilk model 404 verirse listedeki sıradaki modele geçer.

Model adları Google tarafında değişebildiği için sabitler en üstte tanımlıdır.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
from typing import Any, Optional, Sequence

from google import genai
from google.genai import errors, types

log = logging.getLogger("gemini_client")

# --------------------------------------------------------------------------- #
# Model sabitleri (gerekirse buradan değiştirin)
# --------------------------------------------------------------------------- #
# text-embedding-004 Ocak 2026'da kapatıldı; yerine gemini-embedding-001 kullanılır.
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_EMBED_DIM = 768  # 3072 / 1536 / 768 desteklenir; 768 küçük ve yeterli

# Sıralı fallback: ilk model kullanılamazsa (404) sıradakine geçilir.
DEFAULT_LLM_MODELS = ("gemini-3.5-flash", "gemini-2.5-flash")

RETRYABLE_CODES = {429, 500, 502, 503, 504}


def get_api_key() -> str:
    """Önce Colab Secrets, sonra GEMINI_API_KEY ortam değişkeni."""
    try:
        from google.colab import userdata  # type: ignore

        key = userdata.get("GEMINI_API_KEY")
    except ImportError:
        key = os.environ.get("GEMINI_API_KEY")
    except Exception as exc:  # SecretNotFoundError, NotebookAccessError ...
        raise RuntimeError(
            "Colab Secrets'tan GEMINI_API_KEY okunamadı. Sol menüdeki 🔑 Secrets bölümünde "
            "anahtarı 'GEMINI_API_KEY' adıyla ekleyin ve 'Notebook access' anahtarını açın."
        ) from exc
    if not key:
        raise RuntimeError("GEMINI_API_KEY bulunamadı.")
    return key


def _l2_normalize(vec: Sequence[float]) -> list[float]:
    """output_dimensionality < 3072 iken vektörler normalize gelmez; kosinüs için normalize ederiz."""
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    return re.sub(r"\s*```$", "", text).strip()


def _with_retry(fn, *, retries: int = 5, base_delay: float = 3.0):
    """429/5xx hatalarında üstel bekleme (exponential backoff) ile yeniden dener."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            if code in RETRYABLE_CODES and attempt < retries:
                delay = min(60.0, base_delay * (2**attempt)) + random.uniform(0, 1)
                log.warning("API hatası %s - %.1f sn sonra yeniden denenecek (%d/%d)",
                            code, delay, attempt + 1, retries)
                time.sleep(delay)
                continue
            raise


class GeminiClient:
    """Embedding + LLM çağrılarını saran ince katman."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_models: Sequence[str] = DEFAULT_LLM_MODELS,
        embed_model: str = DEFAULT_EMBED_MODEL,
        embed_dim: int = DEFAULT_EMBED_DIM,
    ) -> None:
        self.client = genai.Client(api_key=api_key or get_api_key())
        self.llm_models = list(llm_models)
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self._active_llm: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Embedding
    # ------------------------------------------------------------------ #
    def embed(
        self,
        texts: Sequence[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 25,
        pause: float = 1.0,
    ) -> list[list[float]]:
        """Metinleri batch'ler halinde vektörleştirir (kota dostu: batch arası bekleme)."""
        if not texts:
            return []
        config = types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self.embed_dim
        )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            resp = _with_retry(
                lambda b=batch: self.client.models.embed_content(
                    model=self.embed_model, contents=b, config=config
                ),
                base_delay=5.0,
            )
            if len(resp.embeddings) != len(batch):
                raise RuntimeError("Embedding sayısı girdi sayısıyla uyuşmuyor.")
            vectors.extend(_l2_normalize(e.values) for e in resp.embeddings)
            if start + batch_size < len(texts):
                time.sleep(pause)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text], task_type="RETRIEVAL_QUERY")[0]

    # ------------------------------------------------------------------ #
    # Üretim
    # ------------------------------------------------------------------ #
    def _generate(self, contents: str, config: types.GenerateContentConfig):
        order = list(self.llm_models)
        if self._active_llm in order:  # çalışan modeli öne al
            order.remove(self._active_llm)
            order.insert(0, self._active_llm)

        last_exc: Optional[Exception] = None
        for model in order:
            try:
                resp = _with_retry(
                    lambda m=model: self.client.models.generate_content(
                        model=m, contents=contents, config=config
                    )
                )
                if self._active_llm != model:
                    log.info("Aktif LLM: %s", model)
                self._active_llm = model
                return resp
            except errors.APIError as exc:
                if getattr(exc, "code", None) in (403, 404):
                    log.warning("Model '%s' kullanılamıyor (%s). Sıradaki deneniyor.",
                                model, exc.code)
                    last_exc = exc
                    continue
                raise
        raise RuntimeError(
            f"Hiçbir LLM modeli kullanılamadı: {self.llm_models}. "
            "gemini_client.DEFAULT_LLM_MODELS listesini güncelleyin."
        ) from last_exc

    def generate_text(
        self, prompt: str, system_instruction: Optional[str] = None, temperature: float = 0.7
    ) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction, temperature=temperature
        )
        resp = self._generate(prompt, config)
        return (resp.text or "").strip()

    def generate_json(
        self,
        prompt: str,
        schema: Any,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Any:
        """JSON Mode: `schema` bir pydantic sınıfıysa doğrulanmış nesne döner, değilse dict."""
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        resp = self._generate(prompt, config)

        parsed = getattr(resp, "parsed", None)
        if parsed is not None:
            return parsed

        text = _strip_fences(resp.text or "")
        if isinstance(schema, type) and hasattr(schema, "model_validate_json"):
            return schema.model_validate_json(text)
        return json.loads(text)
