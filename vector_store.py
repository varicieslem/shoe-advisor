"""
vector_store.py
===============
ChromaDB tabanlı ürün deposu.

- Indeksleme: Gemini embedding'leri 25'lik batch'lerle üretilir, her batch hemen diske yazılır
  (kota hatasında kaldığı yerden devam edilebilir).
- PATH A  `filter_search`  : Vektör aramasına HİÇ uğramaz; yalnızca metadata filtresi (`collection.get`).
- PATH B  `semantic_search`: Kosinüs benzerliği ile vektör araması; istenirse metadata ön filtresi ile.
- Beden filtresi: `size_39 ... size_43` Boolean bayrakları (liste operatörlerine bağımlılık yok).
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from gemini_client import GeminiClient

log = logging.getLogger("vector_store")

SIZES = [39, 40, 41, 42, 43]


# --------------------------------------------------------------------------- #
# Filtre modeli
# --------------------------------------------------------------------------- #
@dataclass
class SearchFilters:
    """Router'dan gelen deterministik kısıtlar."""
    size: Optional[int] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    min_rating: Optional[float] = None
    brand: Optional[str] = None
    shoe_types: Optional[Sequence[str]] = None
    gender: Optional[str] = None  # "men" | "women"  (unisex her zaman dahil edilir)
    in_stock_only: bool = True

    def describe(self) -> dict[str, Any]:
        """Boş olmayan kısıtların okunabilir özeti (log / LLM bağlamı için)."""
        raw = {
            "beden": self.size, "max_fiyat_TL": self.max_price, "min_fiyat_TL": self.min_price,
            "min_puan": self.min_rating, "marka": self.brand,
            "tür": list(self.shoe_types) if self.shoe_types else None,
            "cinsiyet": self.gender, "sadece_stokta": self.in_stock_only,
        }
        return {k: v for k, v in raw.items() if v not in (None, False)}


@dataclass
class SearchResult:
    products: list[dict[str, Any]] = field(default_factory=list)
    where: Optional[dict[str, Any]] = None
    notes: list[str] = field(default_factory=list)


def build_where(f: SearchFilters, brand_lower: Optional[str] = None) -> Optional[dict[str, Any]]:
    """SearchFilters -> ChromaDB `where` sözlüğü (birden fazla koşul için $and)."""
    clauses: list[dict[str, Any]] = []
    if f.in_stock_only:
        clauses.append({"in_stock": {"$eq": True}})
    if f.size is not None:
        if f.size not in SIZES:
            raise ValueError(f"Katalogda sadece {SIZES} bedenleri var, istenen: {f.size}")
        clauses.append({f"size_{f.size}": {"$eq": True}})
    if f.max_price is not None:
        clauses.append({"price": {"$lte": float(f.max_price)}})
    if f.min_price is not None:
        clauses.append({"price": {"$gte": float(f.min_price)}})
    if f.min_rating:
        clauses.append({"rating": {"$gte": float(f.min_rating)}})
    if brand_lower:
        clauses.append({"brand_lower": {"$eq": brand_lower}})
    if f.shoe_types:
        types_ = list(f.shoe_types)
        clauses.append(
            {"shoe_type": {"$eq": types_[0]}} if len(types_) == 1
            else {"shoe_type": {"$in": types_}}
        )
    if f.gender:
        clauses.append({"gender": {"$in": [f.gender, "unisex"]}})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def product_to_metadata(p: dict[str, Any]) -> dict[str, Any]:
    """Unified Schema ürününü ChromaDB'nin kabul ettiği düz (str/int/float/bool) metadata'ya çevirir."""
    meta: dict[str, Any] = {
        "asin": p["asin"],
        "title": p["title"][:300],
        "brand": p["brand"],
        "brand_lower": p["brand"].lower(),
        "price": float(p["price"]),
        "price_usd": float(p["price_usd"]),
        "rating": float(p["rating"]),
        "rating_number": int(p["rating_number"]),
        "shoe_type": p["shoe_type"],
        "gender": p["gender"],
        "in_stock": bool(p["in_stock"]),
        "stock_quantity": int(p["stock_quantity"]),
        "colors": ",".join(p["colors"]),
        "sizes_csv": ",".join(str(s) for s in p["sizes"]),
        "raw_json": json.dumps(p, ensure_ascii=False),  # tam ürün kaydı (ayrı DB gerekmez)
    }
    for size in SIZES:  # Boolean beden bayrakları
        meta[f"size_{size}"] = size in p["sizes"]
    return meta


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _quality_score(p: dict[str, Any]) -> float:
    """Az oy alan ürünlerin şişmesini önleyen Bayesian ağırlıklı puan."""
    rating, n = float(p.get("rating", 0)), int(p.get("rating_number", 0))
    prior_mean, prior_weight = 4.0, 20
    return (rating * n + prior_mean * prior_weight) / (n + prior_weight) + 0.05 * math.log1p(n)


# --------------------------------------------------------------------------- #
# Vektör deposu
# --------------------------------------------------------------------------- #
class ShoeVectorStore:
    def __init__(
        self,
        llm: GeminiClient,
        persist_dir: str | Path = "data/chroma",
        collection_name: str = "shoes",
    ) -> None:
        import chromadb

        self.llm = llm
        self.collection_name = collection_name
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self._open_collection()
        self._brand_map: dict[str, str] = {}
        self._refresh_brands()

    # -------------------------- koleksiyon yönetimi -------------------------- #
    def _collection_meta(self) -> dict[str, Any]:
        return {
            "hnsw:space": "cosine",
            "embedding_model": self.llm.embed_model,
            "embedding_dim": self.llm.embed_dim,
        }

    def _create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name, metadata=self._collection_meta()
        )

    def _open_collection(self):
        try:
            col = self.client.get_collection(self.collection_name)
        except Exception:
            return self._create_collection()

        meta = col.metadata or {}
        if (meta.get("embedding_model") != self.llm.embed_model
                or meta.get("embedding_dim") != self.llm.embed_dim):
            log.warning("Embedding modeli/boyutu değişmiş; koleksiyon yeniden oluşturuluyor.")
            self.client.delete_collection(self.collection_name)
            return self._create_collection()
        return col

    def _refresh_brands(self) -> None:
        data = self.collection.get(include=["metadatas"])
        self._brand_map = {
            _normalize(m["brand"]): m["brand_lower"] for m in (data.get("metadatas") or [])
        }

    # ------------------------------- indeksleme ------------------------------ #
    def index_products(
        self,
        products: list[dict[str, Any]],
        batch_size: int = 25,
        pause: float = 1.5,
        rebuild: bool = False,
    ) -> int:
        """Ürünleri vektörleştirip ChromaDB'ye yazar. Mevcut ID'ler atlanır (devam edilebilir)."""
        if rebuild:
            self.client.delete_collection(self.collection_name)
            self.collection = self._create_collection()

        existing = set(self.collection.get(include=[])["ids"])
        todo = [p for p in products if p["asin"] not in existing]
        log.info("Toplam: %d | Zaten indekslenmiş: %d | Yapılacak: %d",
                 len(products), len(existing), len(todo))

        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            vectors = self.llm.embed(
                [p["semantic_text"] for p in batch],
                task_type="RETRIEVAL_DOCUMENT",
                batch_size=batch_size,
            )
            self.collection.upsert(
                ids=[p["asin"] for p in batch],
                embeddings=vectors,
                documents=[p["semantic_text"] for p in batch],
                metadatas=[product_to_metadata(p) for p in batch],
            )
            log.info("İndekslendi: %d / %d", min(start + batch_size, len(todo)), len(todo))
            if start + batch_size < len(todo):
                time.sleep(pause)  # kota dostu bekleme

        self._refresh_brands()
        return self.collection.count()

    # ------------------------------- yardımcılar ------------------------------ #
    @staticmethod
    def _to_product(meta: dict[str, Any]) -> dict[str, Any]:
        return json.loads(meta["raw_json"])

    def list_brands(self) -> list[str]:
        return sorted(set(self._brand_map.values()))

    def resolve_brand(self, name: str) -> Optional[str]:
        """Kullanıcının yazdığı marka adını katalogdaki markaya eşler (büyük/küçük harf, kısmi, bulanık)."""
        target = _normalize(name)
        if not target:
            return None
        if target in self._brand_map:
            return self._brand_map[target]
        for key, value in self._brand_map.items():
            if len(target) >= 3 and len(key) >= 3 and (target in key or key in target):
                return value
        close = difflib.get_close_matches(target, list(self._brand_map), n=1, cutoff=0.8)
        return self._brand_map[close[0]] if close else None

    def _prepare_where(
        self, filters: Optional[SearchFilters]
    ) -> tuple[Optional[dict[str, Any]], list[str], bool]:
        """(where, notlar, devam_edilebilir_mi)"""
        if filters is None:
            return None, [], True
        notes: list[str] = []
        brand_lower = None
        if filters.brand:
            brand_lower = self.resolve_brand(filters.brand)
            if brand_lower is None:
                notes.append(f"'{filters.brand}' markası katalogda bulunamadı.")
                return None, notes, False
        try:
            return build_where(filters, brand_lower), notes, True
        except ValueError as exc:
            notes.append(str(exc))
            return None, notes, False

    # ------------------------- PATH A: deterministik filtre ------------------------- #
    def filter_search(self, filters: SearchFilters, top_k: int = 5) -> SearchResult:
        """Vektör aramasına uğramadan, yalnızca metadata kısıtlarıyla ürün getirir."""
        where, notes, ok = self._prepare_where(filters)
        if not ok:
            return SearchResult([], None, notes)

        data = self.collection.get(where=where, include=["metadatas"], limit=1000)
        products = [self._to_product(m) for m in (data.get("metadatas") or [])]
        products.sort(key=_quality_score, reverse=True)
        return SearchResult(products[:top_k], where, notes)

    # ------------------------- PATH B: semantik arama ------------------------- #
    def semantic_search(
        self, query: str, top_k: int = 5, filters: Optional[SearchFilters] = None
    ) -> SearchResult:
        """Sorguyu embed edip kosinüs benzerliğiyle en yakın ürünleri getirir."""
        where, notes, ok = self._prepare_where(filters)
        if not ok:
            return SearchResult([], None, notes)

        qvec = self.llm.embed_query(query)
        res = self.collection.query(
            query_embeddings=[qvec],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"],
        )
        products: list[dict[str, Any]] = []
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            product = self._to_product(meta)
            product["similarity"] = round(1.0 - float(dist), 4)  # cosine distance -> similarity
            products.append(product)
        return SearchResult(products, where, notes)

    # ---------------------- PATH C yardımcıları: ürün çözümleme ---------------------- #
    def get_by_asin(self, asin: str) -> Optional[dict[str, Any]]:
        data = self.collection.get(ids=[asin], include=["metadatas"])
        metas = data.get("metadatas") or []
        return self._to_product(metas[0]) if metas else None

    def resolve_product(self, mention: str) -> Optional[dict[str, Any]]:
        """Kullanıcının bahsettiği ürünü (ASIN veya serbest metin) katalogdaki ürüne eşler."""
        by_id = self.get_by_asin(mention.strip())
        if by_id:
            return by_id
        found = self.semantic_search(mention, top_k=1).products
        return found[0] if found else None
