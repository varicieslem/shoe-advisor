"""
router.py
=========
Query Intent Router - sistemin kalbi.

Kullanıcı mesajını Gemini + JSON Mode (response_schema) ile analiz eder ve 3 yoldan birini seçer:

  FILTER_SEARCH      (PATH A) -> yalnızca ölçülebilir kısıtlar var; vektör araması YOK.
  SEMANTIC_SEARCH    (PATH B) -> his / kullanım amacı var; vektör araması (+ varsa metadata ön filtresi).
  COMPARE_REASONING  (PATH C) -> iki ürünün karşılaştırılması isteniyor; LLM reasoning.

LLM çağrısı başarısız olursa kural tabanlı (regex) bir yedek yönlendirici devreye girer,
böylece demo hiçbir zaman "router yüzünden" çökmez.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from gemini_client import GeminiClient
from vector_store import SIZES

log = logging.getLogger("router")


# --------------------------------------------------------------------------- #
# Şema (LLM'in döndüreceği JSON)
# --------------------------------------------------------------------------- #
class Intent(str, Enum):
    FILTER_SEARCH = "FILTER_SEARCH"
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"
    COMPARE_REASONING = "COMPARE_REASONING"


class ShoeType(str, Enum):
    running = "running"
    sneaker = "sneaker"
    boot = "boot"
    loafer = "loafer"


class Gender(str, Enum):
    men = "men"
    women = "women"


class RouterDecision(BaseModel):
    intent: Intent = Field(description="Seçilen yol")
    reasoning: str = Field(description="Bu yolun neden seçildiğine dair tek cümlelik Türkçe gerekçe")
    target_size: Optional[int] = Field(description="Türkiye ayakkabı numarası (ör. 42); yoksa null")
    max_price: Optional[float] = Field(description="Üst fiyat sınırı TL cinsinden; yoksa null")
    min_price: Optional[float] = Field(description="Alt fiyat sınırı TL cinsinden; yoksa null")
    brand: Optional[str] = Field(description="Marka adı (ör. Nike); yoksa null")
    shoe_types: Optional[list[ShoeType]] = Field(description="İstenen ayakkabı türleri; yoksa null")
    gender: Optional[Gender] = Field(description="Cinsiyet belirtildiyse men/women; yoksa null")
    min_rating: Optional[float] = Field(description="İstenen minimum puan (1-5); yoksa null")
    semantic_query: Optional[str] = Field(
        description="SEMANTIC_SEARCH için kullanıcının ihtiyacının İngilizce, kısa ve zengin yeniden yazımı; "
                    "diğer yollarda null")
    compare_indices: Optional[list[int]] = Field(
        description="Önceki sonuç listesine 1 tabanlı referanslar (ör. 'ilk ikisi' -> [1, 2]); yoksa null")
    compare_queries: Optional[list[str]] = Field(
        description="Kullanıcının adıyla andığı ürünler (ör. ['Nike Pegasus', 'Adidas Ultraboost']); yoksa null")


SYSTEM_INSTRUCTION = f"""Sen bir e-ticaret ayakkabı mağazasının sorgu yönlendiricisisin.
Kullanıcı mesajını analiz et ve SADECE şemaya uygun JSON üret.

YOL SEÇİMİ
- FILTER_SEARCH: Mesaj yalnızca ölçülebilir kısıtlardan oluşuyor (numara, fiyat, marka, tür, cinsiyet, puan);
  öznel bir ihtiyaç/his YOK. Örnek: "1000 TL altı 42 numara Nike spor ayakkabı".
- SEMANTIC_SEARCH: Kullanım amacı, his veya öznel ihtiyaç var (rahatlık, nefes alma, ayakta çalışma, hafiflik...).
  Mesajda numara/fiyat/marka da varsa bunları slot olarak yine doldur (ön filtre olarak kullanılacak).
  Örnek: "Hastanede bütün gün nöbetteyim, ayağımı ağrıtmayacak, nefes alan bir şey arıyorum".
- COMPARE_REASONING: İki ürünün kıyaslanması isteniyor (fark, hangisi daha iyi...). Ürünler ya adıyla anılır
  (compare_queries) ya da önceki listeye atıfla belirtilir: "ilk ikisi", "1. ile 3." (compare_indices, 1 tabanlı).

SLOT KURALLARI
- Numara: Türkiye numarası (sadece {SIZES} katalogda var). "42 numara", "42 beden" -> 42.
- Fiyat TL'dir. "1500 TL altı" -> max_price=1500. "1.500", "bin beş yüz" -> 1500.
- "spor ayakkabı" -> shoe_types=["sneaker","running"]; "koşu ayakkabısı" -> ["running"]; "bot" -> ["boot"];
  "loafer/mokasen" -> ["loafer"]. Belirtilmemişse null.
- "erkek" -> men, "kadın/bayan" -> women. Belirtilmemişse null.
- semantic_query: Katalog İngilizce olduğu için ihtiyacı İngilizceye çevir ve zenginleştir
  (ör. "comfortable cushioned breathable shoes for standing all day, soft sole, mesh upper").
- Belirtilmeyen her slot null olmalı. Asla uydurma değer yazma."""


# --------------------------------------------------------------------------- #
# Yönlendirme sonucu
# --------------------------------------------------------------------------- #
@dataclass
class RoutedQuery:
    intent: Intent
    reasoning: str
    target_size: Optional[int] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    brand: Optional[str] = None
    shoe_types: Optional[list[str]] = None
    gender: Optional[str] = None
    min_rating: Optional[float] = None
    semantic_query: Optional[str] = None
    compare_indices: list[int] = field(default_factory=list)
    compare_queries: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source: str = "llm"  # "llm" | "fallback"

    def slots(self) -> dict[str, Any]:
        keys = ["target_size", "max_price", "min_price", "brand", "shoe_types", "gender",
                "min_rating", "semantic_query", "compare_indices", "compare_queries"]
        return {k: getattr(self, k) for k in keys if getattr(self, k) not in (None, [], "")}


# --------------------------------------------------------------------------- #
# Doğrulama / temizleme (LLM çıktısına körü körüne güvenme)
# --------------------------------------------------------------------------- #
def _sanitize(decision: RouterDecision, n_previous: int, source: str) -> RoutedQuery:
    warnings: list[str] = []

    size = decision.target_size
    if size is not None and size not in SIZES:
        warnings.append(f"{size} numara katalogda yok (mevcut: {SIZES}); beden kısıtı yok sayıldı.")
        size = None

    max_price = decision.max_price if decision.max_price and decision.max_price > 0 else None
    min_price = decision.min_price if decision.min_price and decision.min_price > 0 else None
    rating = decision.min_rating if decision.min_rating and 0 < decision.min_rating <= 5 else None

    indices = [i for i in (decision.compare_indices or []) if 1 <= i <= n_previous]
    indices = list(dict.fromkeys(indices))  # tekrarları at, sırayı koru
    queries = [q.strip() for q in (decision.compare_queries or []) if q and q.strip()]

    return RoutedQuery(
        intent=decision.intent,
        reasoning=decision.reasoning,
        target_size=size,
        max_price=max_price,
        min_price=min_price,
        brand=(decision.brand or "").strip() or None,
        shoe_types=[t.value for t in decision.shoe_types] if decision.shoe_types else None,
        gender=decision.gender.value if decision.gender else None,
        min_rating=rating,
        semantic_query=(decision.semantic_query or "").strip() or None,
        compare_indices=indices,
        compare_queries=queries,
        warnings=warnings,
        source=source,
    )


# --------------------------------------------------------------------------- #
# Yedek: kural tabanlı yönlendirici (LLM kullanılamadığında)
# --------------------------------------------------------------------------- #
_COMPARE_RE = re.compile(r"karşılaştır|kıyasla|arasındaki fark|farkı ne|hangisi|\bvs\b|\bversus\b", re.I)
_SUBJECTIVE_RE = re.compile(
    r"rahat|yumuşak|nefes|terlet|ağrı|ağrıt|hafif|ayakta|nöbet|yürüyüş|tabanı|konfor|sıcak|soğuk|su geçir",
    re.I)
_SIZE_RE = re.compile(r"\b(3[0-9]|4[0-9])\s*(?:numara|beden|no\b)", re.I)
_PRICE_RE = re.compile(r"(\d[\d.,]*)\s*(?:tl|₺|lira)", re.I)
_ORDINAL_RE = re.compile(r"ilk\s+iki|1\.\s*(?:ile|ve)\s*2\.", re.I)


def _parse_tl(raw: str) -> Optional[float]:
    s = raw.strip().rstrip(".,")
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", s):  # 1.500 veya 1.500,50
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        value = float(s)
    except ValueError:
        return None
    return value if value > 0 else None


def rule_based_route(query: str, n_previous: int = 0) -> RoutedQuery:
    size_match = _SIZE_RE.search(query)
    price_match = _PRICE_RE.search(query)
    size = int(size_match.group(1)) if size_match else None
    price = _parse_tl(price_match.group(1)) if price_match else None
    lowered = query.lower()
    max_price = min_price = None
    if price is not None:
        if re.search(r"üzeri|üstü|en az|fazla", lowered):
            min_price = price
        else:
            max_price = price

    shoe_types = None
    if re.search(r"koşu", lowered):
        shoe_types = ["running"]
    elif re.search(r"spor ayakkab|sneaker", lowered):
        shoe_types = ["sneaker", "running"]

    if _COMPARE_RE.search(query):
        indices = [1, 2] if (_ORDINAL_RE.search(query) and n_previous >= 2) else []
        intent, why = Intent.COMPARE_REASONING, "Karşılaştırma ifadeleri tespit edildi (yedek yönlendirici)."
        semantic = None
    elif _SUBJECTIVE_RE.search(query) or not (size or price):
        indices, semantic = [], query
        intent, why = Intent.SEMANTIC_SEARCH, "Öznel ihtiyaç ifadesi var (yedek yönlendirici)."
    else:
        indices, semantic = [], None
        intent, why = Intent.FILTER_SEARCH, "Sadece ölçülebilir kısıtlar var (yedek yönlendirici)."

    warnings = []
    if size is not None and size not in SIZES:
        warnings.append(f"{size} numara katalogda yok (mevcut: {SIZES}); beden kısıtı yok sayıldı.")
        size = None

    return RoutedQuery(
        intent=intent, reasoning=why, target_size=size, max_price=max_price, min_price=min_price,
        shoe_types=shoe_types, semantic_query=semantic, compare_indices=indices,
        warnings=warnings, source="fallback",
    )


# --------------------------------------------------------------------------- #
# Ana fonksiyon
# --------------------------------------------------------------------------- #
def _previous_results_block(last_results: list[dict[str, Any]]) -> str:
    if not last_results:
        return "Önceki sonuç listesi: (yok)"
    lines = [f"{i}. {p['title'][:80]} ({p['brand']})" for i, p in enumerate(last_results, 1)]
    return "Önceki sonuç listesi (kullanıcı bunlara 'ilk ikisi', '3. ürün' gibi atıf yapabilir):\n" + "\n".join(lines)


def route_query(
    llm: GeminiClient, query: str, last_results: Optional[list[dict[str, Any]]] = None
) -> RoutedQuery:
    """Kullanıcı mesajını analiz edip yol + slot'ları döndürür."""
    last_results = last_results or []
    prompt = f"{_previous_results_block(last_results)}\n\nKullanıcı mesajı: {query}"
    try:
        decision = llm.generate_json(
            prompt, schema=RouterDecision, system_instruction=SYSTEM_INSTRUCTION, temperature=0.0
        )
        if isinstance(decision, dict):  # parsed dict gelirse doğrula
            decision = RouterDecision.model_validate(decision)
        routed = _sanitize(decision, n_previous=len(last_results), source="llm")
        log.info("Router [llm] -> %s | %s", routed.intent.value, routed.slots())
        return routed
    except Exception as exc:  # ağ, kota, şema hatası vb.
        log.warning("LLM router başarısız (%s: %s); kural tabanlı yedek kullanılıyor.",
                    type(exc).__name__, exc)
        return rule_based_route(query, n_previous=len(last_results))
