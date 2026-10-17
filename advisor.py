"""
advisor.py
==========
1. SalesAdvisor            : Arama sonuçlarını bağlam alıp samimi Türkçe satış danışmanı cevabı üretir
                             ve PATH C için yapılandırılmış ürün karşılaştırması yapar.
2. ShoeAdvisorPipeline     : Router -> (A | B | C) -> Danışman akışını uçtan uca yöneten orkestratör.

Halüsinasyon önlemi: LLM'e yalnızca veritabanından gelen alanlar verilir ve
"veride olmayan özelliği uydurma" kuralı prompt'ta açıkça belirtilir.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Optional

from pydantic import BaseModel, Field

from gemini_client import GeminiClient
from router import Intent, RoutedQuery, route_query
from vector_store import SearchFilters, SearchResult, ShoeVectorStore

log = logging.getLogger("advisor")


# --------------------------------------------------------------------------- #
# Biçimlendirme yardımcıları
# --------------------------------------------------------------------------- #
def format_price(value: float) -> str:
    return f"{value:,.0f} TL".replace(",", ".")


def product_context(p: dict[str, Any], idx: int) -> str:
    """Bir ürünü LLM'e verilecek kompakt bağlam metnine çevirir (sadece DB alanları)."""
    specs = {k: v for k, v in (p.get("specs") or {}).items() if k not in ("shoe_type", "gender")}
    spec_text = "; ".join(f"{k}: {str(v)[:60]}" for k, v in list(specs.items())[:8]) or "belirtilmemiş"
    feats = " | ".join(f[:160] for f in (p.get("features") or [])[:5]) or "belirtilmemiş"
    stock = f"stokta ({p['stock_quantity']} adet)" if p.get("in_stock") else "stokta yok"
    sim = f" | benzerlik: {p['similarity']}" if "similarity" in p else ""
    return (
        f"[{idx}] {p['title']}\n"
        f"  Marka: {p['brand']} | Fiyat: {format_price(p['price'])} | "
        f"Puan: {p['rating']:.1f}/5 ({p['rating_number']} değerlendirme){sim}\n"
        f"  Tür: {p['shoe_type']} | Cinsiyet: {p['gender']} | Mevcut numaralar: {p['sizes']} | {stock}\n"
        f"  Özellikler: {feats}\n"
        f"  Teknik detaylar: {spec_text}"
    )


# --------------------------------------------------------------------------- #
# Karşılaştırma çıktısı şeması (PATH C)
# --------------------------------------------------------------------------- #
class CriterionRow(BaseModel):
    criterion: str = Field(description="Karşılaştırma kriteri (ör. Konfor, Dayanıklılık, Havalandırma)")
    product_a: str = Field(description="A ürününün bu kriterdeki durumu; veride yoksa 'veride belirtilmemiş'")
    product_b: str = Field(description="B ürününün bu kriterdeki durumu; veride yoksa 'veride belirtilmemiş'")
    better: str = Field(description="'A', 'B' veya 'Eşit'")


class UseCaseVerdict(BaseModel):
    use_case: str = Field(description="Kullanım senaryosu (ör. Koşu, Gün boyu ayakta çalışma)")
    winner: str = Field(description="'A', 'B' veya 'Berabere'")
    reason: str = Field(description="Verilere dayanan kısa gerekçe")


class ComparisonOutput(BaseModel):
    criteria: list[CriterionRow]
    use_case_verdicts: list[UseCaseVerdict]
    final_advice: str = Field(description="Samimi danışman üslubuyla 2-3 cümlelik net tavsiye")


# --------------------------------------------------------------------------- #
# Satış danışmanı
# --------------------------------------------------------------------------- #
ADVISOR_SYSTEM = """Sen samimi, güvenilir ve bilgili bir ayakkabı mağazası satış danışmanısın.
Türkçe konuş, sıcak ama abartısız ol. Kurallar:
- SADECE sana verilen ürün verilerini kullan. Veride olmayan bir özelliği (malzeme, taban teknolojisi vb.) UYDURMA.
  Bir bilgi yoksa dürüstçe "ürün sayfasında bu konuda ayrıntı yok" de.
- En fazla 3 ürün öner. Her ürün için: adı, fiyatı (TL), 2-3 somut artı ve NEDEN bu kullanıcıya uygun olduğunu yaz.
- Kullanıcının istediği numara/fiyat kısıtlarını ve numaranın mevcut olduğunu belirt.
- Fiyat/puan/numara bilgilerini aynen verilerdeki gibi kullan.
- Sonda tek bir kısa devam sorusu sor (ör. bütçe, kullanım sıklığı).
- Madde işaretini ölçülü kullan; metin sohbet havasında olsun."""

COMPARE_SYSTEM = """Sen tarafsız ve dürüst bir ayakkabı danışmanısın.
Sana A ve B ürünlerinin veritabanı kayıtları verilecek. Kurallar:
- SADECE verilen verileri kullan; veride olmayan özelliği uydurma. Veri yetersizse 'veride belirtilmemiş' yaz.
- 4-6 kriter seç (konfor, havalandırma, dayanıklılık, kullanım alanı, fiyat/performans gibi), yalnızca veriyle
  desteklenebilenleri kullan.
- Kullanıcının sorduğu kullanım senaryosunu (ör. koşu) use_case_verdicts içine mutlaka ekle.
- Kazananı dürüstçe seç; belirsizse 'Eşit' / 'Berabere' de."""


class SalesAdvisor:
    def __init__(self, llm: GeminiClient) -> None:
        self.llm = llm

    # ---------------------- PATH A / B cevabı ---------------------- #
    def recommend(
        self,
        user_query: str,
        products: list[dict[str, Any]],
        notes: Optional[list[str]] = None,
        applied_filters: Optional[dict[str, Any]] = None,
    ) -> str:
        context = "\n\n".join(product_context(p, i) for i, p in enumerate(products[:5], 1))
        extra = ""
        if notes:
            extra += "\nSistem notları (kullanıcıya şeffafça belirt): " + " ".join(notes)
        if applied_filters:
            extra += f"\nUygulanan filtreler: {applied_filters}"
        prompt = (
            f"Kullanıcı isteği: {user_query}\n{extra}\n\n"
            f"Veritabanından gelen ürünler:\n{context}\n\n"
            "Bu ürünlerden kullanıcıya en uygun olanları öner."
        )
        try:
            return self.llm.generate_text(prompt, system_instruction=ADVISOR_SYSTEM, temperature=0.6)
        except Exception as exc:
            log.warning("Danışman LLM başarısız (%s); statik cevap üretiliyor.", exc)
            return self.fallback_answer(products, notes)

    @staticmethod
    def fallback_answer(products: list[dict[str, Any]], notes: Optional[list[str]] = None) -> str:
        lines = ["Şu an sana en uygun bulduklarım:"]
        for p in products[:3]:
            lines.append(f"- {p['title'][:90]} ({p['brand']}) - {format_price(p['price'])}, "
                         f"puan {p['rating']:.1f}/5, numaralar: {p['sizes']}")
        if notes:
            lines.append("Not: " + " ".join(notes))
        return "\n".join(lines)

    # ---------------------- PATH C: karşılaştırma ---------------------- #
    @staticmethod
    def facts_table(a: dict[str, Any], b: dict[str, Any]) -> str:
        """Deterministik (LLM'siz) temel veri tablosu: halüsinasyon riski sıfır."""
        def short(p: dict[str, Any]) -> str:
            return p["title"][:38] + ("…" if len(p["title"]) > 38 else "")

        rows = [
            ("Marka", a["brand"], b["brand"]),
            ("Fiyat", format_price(a["price"]), format_price(b["price"])),
            ("Puan", f"{a['rating']:.1f} ({a['rating_number']})", f"{b['rating']:.1f} ({b['rating_number']})"),
            ("Tür", a["shoe_type"], b["shoe_type"]),
            ("Numaralar", ", ".join(map(str, a["sizes"])), ", ".join(map(str, b["sizes"]))),
            ("Stok", "var" if a["in_stock"] else "yok", "var" if b["in_stock"] else "yok"),
        ]
        out = [f"| Özellik | A: {short(a)} | B: {short(b)} |", "|---|---|---|"]
        out += [f"| {k} | {x} | {y} |" for k, x, y in rows]
        return "\n".join(out)

    def compare(self, user_query: str, a: dict[str, Any], b: dict[str, Any]) -> str:
        prompt = (
            f"Kullanıcı sorusu: {user_query}\n\n"
            f"A ürünü:\n{product_context(a, 1)}\n\nB ürünü:\n{product_context(b, 2)}"
        )
        header = "### Temel veriler\n" + self.facts_table(a, b)
        try:
            out: ComparisonOutput = self.llm.generate_json(
                prompt, schema=ComparisonOutput, system_instruction=COMPARE_SYSTEM, temperature=0.2
            )
            if isinstance(out, dict):
                out = ComparisonOutput.model_validate(out)
        except Exception as exc:
            log.warning("Karşılaştırma LLM başarısız (%s).", exc)
            return header + "\n\nŞu an ayrıntılı karşılaştırma üretemedim, lütfen tekrar dene."

        crit = ["| Kriter | A | B | Daha iyi |", "|---|---|---|---|"]
        crit += [f"| {r.criterion} | {r.product_a} | {r.product_b} | {r.better} |" for r in out.criteria]
        verdicts = "\n".join(f"- **{v.use_case}** -> {v.winner}: {v.reason}" for v in out.use_case_verdicts)
        return (f"{header}\n\n### Teknik kıyaslama\n" + "\n".join(crit)
                + f"\n\n### Kullanım senaryosuna göre\n{verdicts}\n\n**Tavsiyem:** {out.final_advice}")


# --------------------------------------------------------------------------- #
# Orkestratör
# --------------------------------------------------------------------------- #
def filters_from_route(r: RoutedQuery, include_types: bool) -> SearchFilters:
    return SearchFilters(
        size=r.target_size, max_price=r.max_price, min_price=r.min_price, min_rating=r.min_rating,
        brand=r.brand, shoe_types=r.shoe_types if include_types else None, gender=r.gender,
        in_stock_only=True,
    )


class ShoeAdvisorPipeline:
    """Kullanıcı mesajı -> Router -> ilgili yol -> Danışman cevabı."""

    # Sonuç çıkmazsa sırayla gevşetilecek (esnek) kriterler. Beden/fiyat/stok asla gevşetilmez.
    RELAXABLE = ("min_rating", "shoe_types", "gender", "brand")

    def __init__(self, llm: GeminiClient, store: ShoeVectorStore, top_k: int = 5) -> None:
        self.llm, self.store, self.top_k = llm, store, top_k
        self.advisor = SalesAdvisor(llm)
        self.last_results: list[dict[str, Any]] = []  # "ilk ikisini karşılaştır" gibi atıflar için

    def ask(self, query: str) -> dict[str, Any]:
        routed = route_query(self.llm, query, self.last_results)
        if routed.intent == Intent.FILTER_SEARCH:
            out = self._path_filter(query, routed)
        elif routed.intent == Intent.SEMANTIC_SEARCH:
            out = self._path_semantic(query, routed)
        else:
            out = self._path_compare(query, routed)

        out.update(intent=routed.intent.value, reasoning=routed.reasoning,
                   slots=routed.slots(), router_source=routed.source)
        out["notes"] = routed.warnings + out.get("notes", [])
        if out.get("products") and routed.intent != Intent.COMPARE_REASONING:
            self.last_results = out["products"]  # karşılaştırma, listeyi bozmasın
        return out

    # ---------------- PATH A ---------------- #
    def _path_filter(self, query: str, r: RoutedQuery) -> dict[str, Any]:
        filters = filters_from_route(r, include_types=True)
        result = self.store.filter_search(filters, top_k=self.top_k)
        notes = list(result.notes)

        # Sonuç yoksa esnek kriterleri tek tek gevşet (ve kullanıcıya söyle).
        if not result.products and not result.notes:
            for name in self.RELAXABLE:
                if getattr(filters, name):
                    filters = replace(filters, **{name: None})
                    result = self.store.filter_search(filters, top_k=self.top_k)
                    if result.products:
                        notes.append(f"Tam eşleşme bulunamadı; '{name}' kriteri gevşetilerek arandı.")
                        break

        if not result.products:
            msg = ("Bu kriterlere uyan stokta ürün bulamadım. "
                   + " ".join(notes + r.warnings)
                   + " Bütçeyi biraz artırmayı veya başka bir numarayı denemek ister misin?")
            return {"products": [], "answer": msg.strip(), "notes": notes,
                    "applied_filters": filters.describe()}

        answer = self.advisor.recommend(query, result.products, notes, filters.describe())
        return {"products": result.products, "answer": answer, "notes": notes,
                "applied_filters": filters.describe()}

    # ---------------- PATH B ---------------- #
    def _path_semantic(self, query: str, r: RoutedQuery) -> dict[str, Any]:
        filters = filters_from_route(r, include_types=False)
        sem_query = r.semantic_query or query
        result: SearchResult = self.store.semantic_search(sem_query, top_k=self.top_k, filters=filters)
        notes = list(result.notes)

        if not result.products:
            msg = ("İhtiyacına uygun, kısıtlarına da uyan ürün bulamadım. "
                   + " ".join(notes + r.warnings)
                   + " Kısıtları biraz esnetelim mi?")
            return {"products": [], "answer": msg.strip(), "notes": notes,
                    "semantic_query": sem_query, "applied_filters": filters.describe()}

        answer = self.advisor.recommend(query, result.products, notes, filters.describe())
        return {"products": result.products, "answer": answer, "notes": notes,
                "semantic_query": sem_query, "applied_filters": filters.describe()}

    # ---------------- PATH C ---------------- #
    def _resolve_compare_targets(self, r: RoutedQuery) -> list[dict[str, Any]]:
        targets: list[dict[str, Any]] = []
        for i in r.compare_indices:
            targets.append(self.last_results[i - 1])
        for mention in r.compare_queries:
            found = self.store.resolve_product(mention)
            if found:
                targets.append(found)
        unique: dict[str, dict[str, Any]] = {}
        for p in targets:  # aynı ürünü iki kez karşılaştırma
            unique.setdefault(p["asin"], p)
        return list(unique.values())

    def _path_compare(self, query: str, r: RoutedQuery) -> dict[str, Any]:
        targets = self._resolve_compare_targets(r)
        if len(targets) < 2:
            msg = ("Karşılaştırmak için iki ürüne ihtiyacım var. Ürün adlarını yazabilir ya da "
                   "önce bir arama yapıp 'ilk ikisini karşılaştır' diyebilirsin.")
            return {"products": [], "answer": msg, "notes": []}
        a, b = targets[0], targets[1]
        answer = self.advisor.compare(query, a, b)
        return {"products": [a, b], "answer": answer, "notes": []}
